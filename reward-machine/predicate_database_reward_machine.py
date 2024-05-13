import duckdb
import numpy as np
import pandas as pd
import pathlib
import tatsu.ast
import tatsu.grammars
import typing
import sys

sys.path.append((pathlib.Path(__file__).parents[1].resolve() / 'src').as_posix())

import ast_printer
import ast_parser
from ast_parser import ASTParser, SECTION_CONTEXT_KEY, VARIABLES_CONTEXT_KEY, update_context_variables, SETUP
from ast_utils import simplified_context_deepcopy, deepcopy_ast, ASTCopyType, replace_child

from compile_predicate_statistics_full_database import CommonSensePredicateStatisticsFullDatabase


DEFAULT_UNSUPPORTED_RULES = [
    'function_comparison',
    'function_eval',
    'predicate_adjacent_side_3',
    'predicate_adjacent_side_4',
    'predicate_between',
    'predicate_faces',
    'predicate_is_setup_object',
    'predicate_opposite',
    'predicate_rug_color_under',
    'predicate_same_color',
    'predicate_same_object',
    'predicate_same_type',
    'super_predicate_exists',
    'super_predicate_forall',
    'once_measure',
    'while_hold',
]


def _pref_forall_pos_to_key(pos: int):
    return f'pref_forall_{pos}'


class MixedTraceFilterGameParser(ASTParser):
    unsupported_rules: typing.Set[str]

    def __init__(self, unsupported_rules: typing.Sequence[str] = DEFAULT_UNSUPPORTED_RULES):
        super().__init__()
        self.expected_keys = set()
        self.unsupported_rules = set(unsupported_rules)

    def __call__(self, ast, **kwargs):
        initial_call = 'inner_call' not in kwargs or not kwargs['inner_call']
        if initial_call:
            kwargs['inner_call'] = True
            kwargs['local_context'] = {'mapping': {VARIABLES_CONTEXT_KEY: {}}}
            kwargs['global_context'] = {}
            self.expected_keys = set()
            self.unsupported_keys = set()
            # self.traces_by_preference_or_section = {}
            # self.preferences_or_sections_with_implemented_predicates = set()
            # self.predicate_strings_by_preference_or_section = defaultdict(set)
            # self.not_implemented_predicate_counts = defaultdict(int)

        retval = super().__call__(ast, **kwargs)

        if initial_call:
            return self.unsupported_keys, self.expected_keys
        else:
            return retval

    def _current_ast_to_contexts_hook(self, ast: tatsu.ast.AST, kwargs: typing.Dict[str, typing.Any]):
        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore

        if rule == 'pref_forall':
            kwargs['local_context']['current_pref_forall_index'] = ast.parseinfo.pos

        if rule == 'preference':
            kwargs['local_context']['current_preference_name'] = ast.pref_name

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        self._current_ast_to_contexts(ast, **kwargs)
        kwargs['local_context']['mapping'] = update_context_variables(ast, kwargs['local_context']['mapping'])

        current_key = None
        if SECTION_CONTEXT_KEY in kwargs and kwargs[SECTION_CONTEXT_KEY] == SETUP:
            current_key = kwargs[SECTION_CONTEXT_KEY]
        elif 'current_pref_forall_index' in kwargs['local_context']:
            current_key =_pref_forall_pos_to_key(kwargs['local_context']['current_pref_forall_index'])
        elif 'current_preference_name' in kwargs['local_context']:
            current_key = kwargs['local_context']['current_preference_name']

        if current_key is not None:
            self.expected_keys.add(current_key)

            if ast.parseinfo.rule in self.unsupported_rules:
                self.unsupported_keys.add(current_key)

        for key in ast:
            if key != 'parseinfo':
                child_kwargs = simplified_context_deepcopy(kwargs)
                retval = self(ast[key], **child_kwargs)
                self._update_contexts_from_retval(kwargs, retval)


DUMMY_PREFERENCE_GAME = """(define (game dummy-preference-game) (:domain many-objects-room-v1)
(:constraints (and
    (preference dummyPreference
            (at-end (game-over))
    )
))
(:scoring (count dummyPreference)
))
"""


class ASTTraceFilterSplitter(ASTParser):
    keep_keys: typing.Set[str]
    remove_keys: typing.Set[str]
    should_insert_dummy_preference: bool

    def __init__(self, grammar_parser: tatsu.grammars.Grammar):
        self.grammar_parser = grammar_parser

    def __call__(self, ast, **kwargs):
        initial_call = 'inner_call' not in kwargs or not kwargs['inner_call']
        if initial_call:
            kwargs['inner_call'] = True

            if 'remove_keys' not in kwargs:
                raise ValueError('remove_keys must be specified')
            self.remove_keys = kwargs['remove_keys']

            if len(self.remove_keys) == 0:
                raise ValueError('remove_keys must be non-empty')

            if 'keep_keys' not in kwargs:
                raise ValueError('keep_keys must be specified')
            self.keep_keys = kwargs['keep_keys']

            if len(self.keep_keys) == 0:
                raise ValueError('keep_keys must be non-empty')

            ast = deepcopy_ast(ast)

            # Handle the setup right here and now, if we're removing it
            if SETUP in self.remove_keys:
                ast = (*ast[:3], *ast[4:])
                # If the only thin we're removing is the setup, we're done
                if len(self.remove_keys) == 1:
                    return ast

            # check if we're only keeping the setup and inserting a dummy preference, because if so, we're done
            if len(self.keep_keys) == 1 and SETUP in self.keep_keys:
                dummy_preference_game = self.grammar_parser.parse(DUMMY_PREFERENCE_GAME)
                return (*ast[:4], dummy_preference_game[3], *ast[4:])

        super().__call__(ast, **kwargs)

        if initial_call:
            return ast

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        rule = ast.parseinfo.rule

        if rule == 'preferences':
            if isinstance(ast.preferences, tatsu.ast.AST):
                raise ValueError(f'If removing a single preference, the initial call should handle it, so this should never occur')

            new_children = typing.cast(typing.List[tatsu.ast.AST], deepcopy_ast(ast.preferences, ASTCopyType.NODE))
            indices_to_remove = []
            for i, child in enumerate(new_children):
                child_def = child.definition
                if child_def.parseinfo.rule == 'preference' and child_def.pref_name in self.remove_keys:
                    print(f'Removing preference {child_def.pref_name}')
                    indices_to_remove.append(i)
                elif child_def.parseinfo.rule == 'pref_forall' and _pref_forall_pos_to_key(child_def.parseinfo.pos) in self.remove_keys:
                    print(f'Removing pref_forall {_pref_forall_pos_to_key(child_def.parseinfo.pos)}')
                    indices_to_remove.append(i)

            for i in reversed(indices_to_remove):
                new_children.remove(new_children[i])

            replace_child(ast, 'preferences', new_children)

        else:
            for key in ast:
                if key != 'parseinfo':
                    self(ast[key], **kwargs)


PREDICATE_IN_DATA_RULE_TO_CHILD = {
    'setup_game_conserved': 'conserved_pred',
    'setup_game_optional': 'optional_pred',
    'once': 'once_pred',
    'once_measure': 'once_measure_pred',
    'hold': 'hold_pred',
    'while_hold': 'hold_pred',
    'at_end': 'at_end_pred',
}
FULL_DATASET_TRACES_HASH = '028b3733'


def _df_intervals_to_array(row):
    return np.unpackbits(np.frombuffer(row['intervals'], dtype=np.uint8))[-row['trace_length']:]


class PredicateDatabaseRewardMachine(ASTParser):
    predicate_data_estimator: CommonSensePredicateStatisticsFullDatabase
    rule_to_child_name_mapping: typing.Dict[str, str]
    trace_names_hash: str

    def __init__(self, trace_names_hash: str = FULL_DATASET_TRACES_HASH, rule_to_child_name_mapping: typing.Dict[str, str] = PREDICATE_IN_DATA_RULE_TO_CHILD):
        self.trace_names_hash = trace_names_hash
        self.rule_to_child_name_mapping = rule_to_child_name_mapping

        self.predicate_data_estimator = CommonSensePredicateStatisticsFullDatabase(
            force_trace_names_hash=self.trace_names_hash
        )

        self.trace_id_to_length_df = self.predicate_data_estimator.con.execute('SELECT * FROM trace_length_and_domains;').fetchdf()
        self.trace_id_to_length_df.drop(columns=['domain'], inplace=True)
        self.trace_id_to_length_df.rename(columns=dict(length='trace_length'), inplace=True)

    def __call__(self, ast, **kwargs):
        initial_call = 'inner_call' not in kwargs or not kwargs['inner_call']
        if initial_call:
            kwargs['inner_call'] = True
            kwargs['local_context'] = {'mapping': {VARIABLES_CONTEXT_KEY: {}}}
            kwargs['global_context'] = {}
            self.satisfactions_by_key = {}
            self.temporary_results_by_key = {}

        retval = super().__call__(ast, **kwargs)

        if initial_call:
            if SETUP in self.temporary_results_by_key:
                self.satisfactions_by_key[SETUP] = self.temporary_results_by_key[SETUP].agg(np.logical_or.reduce).sum()

            return self.satisfactions_by_key
        else:
            return retval

    def _current_ast_to_contexts_hook(self, ast: tatsu.ast.AST, kwargs: typing.Dict[str, typing.Any]):
        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore

        if rule == 'pref_forall':
            kwargs['local_context']['current_pref_forall_index'] = ast.parseinfo.pos

        if rule == 'preference':
            kwargs['local_context']['current_preference_name'] = ast.pref_name

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        self._current_ast_to_contexts(ast, **kwargs)
        kwargs['local_context']['mapping'] = update_context_variables(ast, kwargs['local_context']['mapping'])

        current_key = None
        if SECTION_CONTEXT_KEY in kwargs and kwargs[SECTION_CONTEXT_KEY] == SETUP:
            current_key = kwargs[SECTION_CONTEXT_KEY]
        elif 'current_pref_forall_index' in kwargs['local_context']:
            current_key =_pref_forall_pos_to_key(kwargs['local_context']['current_pref_forall_index'])
        elif 'current_preference_name' in kwargs['local_context']:
            current_key = kwargs['local_context']['current_preference_name']

        if current_key == SETUP:
            if ast.parseinfo.rule in self.rule_to_child_name_mapping:
                pred = ast[self.rule_to_child_name_mapping[ast.parseinfo.rule]]
                results_df = self.predicate_data_estimator.filter(pred, mapping, use_de_morgans=True)
                results_df = results_df.assign(intervals=results_df.merge(self.trace_id_to_length_df, on=['trace_id'], how='left').apply(_df_intervals_to_array, axis=1))
                setup_intervals_series = results_df.groupby('trace_id', as_index=True, observed=True).intervals.agg(lambda x: reduce(np.logical_or, x.values).astype(np.uint8))

                if current_key not in self.temporary_results_by_key:
                    self.temporary_results_by_key[current_key] = setup_intervals_series

                else:
                    merged = pd.merge(self.temporary_results_by_key[current_key], setup_intervals_series, left_index=True, right_index=True, how='inner')
                    self.temporary_results_by_key[current_key] = merged.agg(lambda row: np.logical_and(row['intervals_x'], row['intervals_y']), axis=1)


        # TODO: next: if we find a preference, check if it's an at-end or a then, and implement the logic accordingly

        if current_key is not None:
            if ast.parseinfo.rule in self.rule_to_child_name_mapping:
                pred = ast[self.rule_to_child_name_mapping[ast.parseinfo.rule]]





if __name__ == '__main__':
    from ast_utils import cached_load_and_parse_games_from_file
    grammar = open('./dsl/dsl.ebnf').read()
    grammar_parser = tatsu.compile(grammar)
    game_asts = list(cached_load_and_parse_games_from_file('./dsl/interactive-beta.pddl', grammar_parser, False, relative_path='.'))

    game_parser = MixedTraceFilterGameParser()
    game_splitter = ASTTraceFilterSplitter(grammar_parser)  # type: ignore
    ast = game_asts[1]
    unsupported, expected = game_parser(ast)
    supported = expected - unsupported

    if len(supported) > 0 and len(unsupported) > 0:
        print(f'Game {ast[1].game_name} has supported keys: {list(supported)} and unsupported keys: {list(unsupported)}')
        supported_only = game_splitter(ast, keep_keys=supported, remove_keys=unsupported)
        unsupported_only = game_splitter(ast, keep_keys=unsupported, remove_keys=supported)

        print('=' * 80)
        print(ast_printer.ast_to_string(supported_only, '\n'))
        print('=' * 80)
        print(ast_printer.ast_to_string(unsupported_only, '\n'))
