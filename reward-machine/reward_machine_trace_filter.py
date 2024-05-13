import argparse
from collections import defaultdict, Counter
import cachetools
from datetime import datetime, date, timedelta
import duckdb
import inspect
import io
import json
import logging
import multiprocessing
import multiprocessing.pool
import multiprocessing.reduction
import networkx as nx
import operator
import os
import shutil
import sys
import tatsu.ast
import traceback
from tqdm import tqdm
import typing
import pathlib
import polars as pl


import compile_predicate_statistics_full_database
from compile_predicate_statistics_full_database import MissingVariableException, PredicateNotImplementedException, DEFAULT_BASE_TRACE_PATH
from config import OBJECTS_BY_ROOM_AND_TYPE, SPECIFIC_NAMED_OBJECTS_BY_ROOM
from game_handler import GameHandler

from utils import FullState


sys.path.append((pathlib.Path(__file__).parents[1].resolve() / 'src').as_posix())
import ast_printer
import ast_parser
from ast_utils import simplified_context_deepcopy, deepcopy_ast, ASTCopyType
from fitness_energy_utils import load_data, save_data, load_data_from_path, save_data_to_path
from ast_parser import SECTION_CONTEXT_KEY, VARIABLES_CONTEXT_KEY
from evolutionary_sampler import *
from fitness_features import PREDICATE_IN_DATA_RULE_TO_CHILD, BOOLEAN_PARSER
from latest_model_paths import MAP_ELITES_MODELS


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.getLogger('wandb').setLevel(logging.WARNING)
logging.getLogger('wandb.docker').setLevel(logging.WARNING)
logging.getLogger('wandb.docker.auth').setLevel(logging.WARNING)


parser = argparse.ArgumentParser()
parser.add_argument('--trace-names-hash', type=str, default=FULL_DATASET_TRACES_HASH)
parser.add_argument('--map-elites-model-name', type=str, default=None)
parser.add_argument('--map-elites-model-date-id', type=str, default=None)
parser.add_argument('--map-elites-run-year', type=str, default=str(date.today().year))
parser.add_argument('--run-from-real-games', action='store_true')
DEFAULT_MODEL_FOLDER = 'samples'
parser.add_argument('--map-elites-model-folder', type=str, default=DEFAULT_MODEL_FOLDER)
DEFAULT_RELATIVE_PATH = '.'
parser.add_argument('--relative-path', type=str, default=DEFAULT_RELATIVE_PATH)
parser.add_argument('--base-trace-path', type=str, default=DEFAULT_BASE_TRACE_PATH)
parser.add_argument('--single-key', type=str, default=None)
DEFAULT_STOP_AFTER_COUNT = 5
parser.add_argument('--stop-after-count', type=int, default=DEFAULT_STOP_AFTER_COUNT)
parser.add_argument('--max-traces-per-game', type=int, default=None)
parser.add_argument('--max-keys', type=int, default=None)
parser.add_argument('--tqdm', action='store_true')
parser.add_argument('--worker-tqdm', action='store_true')
parser.add_argument('--use-trace-intersection', action='store_true')
parser.add_argument('--use-only-database-nonconfirmed-traces', action='store_true')
parser.add_argument('--recompute-traces-by-population-key', action='store_true')
parser.add_argument('--n-workers', type=int, default=1)
parser.add_argument('--chunksize', type=int, default=1)
parser.add_argument('--maxtasksperchild', default=None, type=int)
DEFAULT_N_DAYS_BACK = 3
parser.add_argument('--n-days-back', type=int, default=DEFAULT_N_DAYS_BACK)
DEFAULT_MAX_TIME_PER_SAMPLE_HOURS = 4.0
parser.add_argument('--max-time-per-sample-hours', type=float, default=DEFAULT_MAX_TIME_PER_SAMPLE_HOURS)
parser.add_argument('--copy-traces-to-tmp', action='store_true')
DEFAULT_START_METHOD = 'spawn'
parser.add_argument('--parallel-start-method', type=str, default=DEFAULT_START_METHOD)
DEFAULT_SAVE_INTERVAL = 8
parser.add_argument('--save-interval', type=int, default=DEFAULT_SAVE_INTERVAL)
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
parser.add_argument('-t', '--test-file', default='./dsl/interactive-beta.pddl')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--force-trace-ids', type=str, nargs='*', default=[], required=False)
parser.add_argument('--dont-sort-keys-by-traces', action='store_true')
TRACES_BY_POPULATION_KEY_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'caches/traces_by_population_key_cache.pkl.gz')
parser.add_argument('--traces-by-population-key-cache-path', type=str, default=TRACES_BY_POPULATION_KEY_CACHE_PATH)



FULL_DATASET_TRACES_HASH = '028b3733'
WILDCARD = '*'
EMPTY_SET = '∅'
# TODO: consuder adding support to `while_hold`
FULL_PREDICATE_STATE_MACHINE_SUPPORTED_MODALS = set(('once', 'once_measure', 'hold', 'while_hold'))
MERGE_IGNORE_COLUMNS = set(['domain', 'intervals'])
LOGIC_EVALUATOR_MAX_MARGIN_STEPS = 1

class PreferenceStateMachineLogicEvaluator:
    trace_id_to_length: typing.Dict[str, int]
    def __init__(self, trace_id_to_length: typing.Dict[str, int], max_margin_steps: int = LOGIC_EVALUATOR_MAX_MARGIN_STEPS):
        self.trace_id_to_length = trace_id_to_length
        self.max_margin_steps = max_margin_steps

    def _intervals_to_strings_apply(self, row: pd.Series):
        length = self.trace_id_to_length[row.trace_id]
        bin_str = bin(int.from_bytes(row.intervals, 'big'))
        # If it's at least the right length, truncate the leading garbage bits, otherwise, truncate only the leading 0b
        start_index = -length if len(bin_str) >= length + 2 else 2
        # Add any missing leading zeros
        return bin_str[start_index:].zfill(length)

    def _intervals_to_integers_apply(self, row: pd.Series):
        # Casting to a string to make sure I truncate the the leading garbage bits
        return int(self._intervals_to_strings_apply(row), 2)

    def _preprocess_results(self, result: typing.Union[pd.DataFrame, typing.Literal['WILDCARD']], to_strings: bool = True):
        if isinstance(result, pd.DataFrame):
            result = result.drop(columns=['domain'])

            if to_strings:
                out = result.apply(self._intervals_to_strings_apply, axis=1)
                if out.size != result.intervals.size:
                    print(f'Expected {result.intervals.size} results, got {out.size}')
                result.intervals = out

            # With df.apply, this ends up trying to infer the dtype as ints and overflowing, so we do this instead:
            else:
                result.intervals = pd.Series((self._intervals_to_integers_apply(row) for row in result.itertuples()), dtype='object')  # type: ignore

        return result

    def _create_wildcard_intervals(self, trace_id: str):
        return '1' * self.trace_id_to_length[trace_id]

    def _state_machine_logic_apply(
            self, row: pd.Series, modals: typing.List[str], trace_ids_found: typing.Set[str],
            check_all_unique_objects_per_trace: bool = False):

        if row.trace_id in trace_ids_found and not check_all_unique_objects_per_trace:
            return True

        index = 0
        state = 0
        margin_steps = 0
        interval_keys = sorted([key for key in row.keys() if key.startswith('intervals')])
        final_state = len(interval_keys)
        intervals = [row[key] for key in interval_keys]

        try:
            while index < self.trace_id_to_length[row.trace_id] and state < final_state:
                next_valid = intervals[state][index] == '1'
                if next_valid:
                    state += 1

                elif state > 0 and (state == 1 or modals[state - 1] == 'hold'):
                    current_valid = intervals[state - 1][index] == '1'
                    if not current_valid:
                        margin_steps += 1

                        if margin_steps > self.max_margin_steps:
                            state = 0
                            margin_steps = 0

                index += 1

        except IndexError:
            print(f'Trace id {row.trace_id} has length {self.trace_id_to_length[row.trace_id]}, but intervals {[len(i) for i in intervals]} for keys {interval_keys} are too short')
            raise

        result = state == final_state
        if result and not check_all_unique_objects_per_trace:
            trace_ids_found.add(row.trace_id)

        return result

    def evaluate_then(self,
                      predicate_results: typing.List[typing.Union[pd.DataFrame, typing.Literal['WILDCARD']]],
                      predicate_modals: typing.List[str],
                      additional_constraint_specifications: typing.Optional[typing.List[typing.Tuple[str, typing.List[str], bool, typing.Union[str, typing.Set[str]]]]]):

        if any((isinstance(result, str) and result == EMPTY_SET) or (isinstance(result, pd.DataFrame) and result.empty)
               for result in predicate_results):
            return []

        preprocessed_results = [self._preprocess_results(result) for result in predicate_results]
        merged_df = None
        wildcard_indices = []
        merged_indices = []

        initial_intervals_index = None
        for i, result in enumerate(preprocessed_results):
            if isinstance(result, pd.DataFrame):
                if merged_df is None:
                    merged_df = result
                    initial_intervals_index = i
                else:
                    merge_columns = ['trace_id']
                    merge_columns.extend(set(merged_df.columns) & set(result.columns) - MERGE_IGNORE_COLUMNS)
                    merged_df = merged_df.merge(result, on=merge_columns, how='inner', suffixes=('', f'_{i + 1}'))
                    merged_indices.append(i)
            else:
                wildcard_indices.append(i)

        if merged_df is None:
            raise ValueError('No non-wildcard results found in then evaluation')

        if initial_intervals_index is None:
            raise ValueError('No initial intervals index found in then evaluation')

        merged_df.rename(columns={'intervals': f'intervals_{initial_intervals_index + 1}'}, inplace=True)

        if additional_constraint_specifications is not None and len(additional_constraint_specifications) > 0:
            merged_df = self._handle_additional_constraints(merged_df, additional_constraint_specifications)

        for i in wildcard_indices:
            merged_df[f'intervals_{i + 1}'] = merged_df.trace_id.apply(self._create_wildcard_intervals)

        state_machine_results = merged_df.apply(self._state_machine_logic_apply, axis=1, result_type='reduce',
                                                modals=predicate_modals, trace_ids_found=set())

        if isinstance(state_machine_results, pd.DataFrame):
            print(f'Expected state machine results to be a series, got a dataframe with columns: {state_machine_results.columns}')

        return merged_df.trace_id[state_machine_results].unique().tolist()

    def _handle_additional_constraints(self, merged_df: pd.DataFrame, additional_constraint_specifications: typing.List[typing.Tuple[str, typing.List[str], bool, typing.Union[str, typing.Set[str]]]]):
        for (rule, args, negated, context_variables) in additional_constraint_specifications:
            row_filter = None
            if rule == 'predicate_same_object':
                if args[0] in merged_df.columns and args[1] in merged_df.columns:
                    row_filter = merged_df[args[0]] == merged_df[args[1]]

                # TODO: theoretically we could construct all combinations that work, and add them to the df
                # but that seems drastically excessive, and we already rule out empty sets in the main trace finder
                else:
                    continue

            else:   # rule == 'predicate_same_type':
                argument_types = typing.cast(typing.List[typing.List[str]], [ast_parser.predicate_function_term_to_types(arg, context_variables) for arg in args])  # type: ignore
                argument_objects = [_argument_types_to_object_ids(arg_types) for arg_types in argument_types]
                shared_object_ids = argument_objects[0] & argument_objects[1]

                if args[0] in merged_df.columns:
                    row_filter = merged_df[args[0]].isin(shared_object_ids)

                if args[1] in merged_df.columns:
                    if row_filter is None:
                        row_filter = merged_df[args[1]].isin(shared_object_ids)
                    else:
                        row_filter &= merged_df[args[1]].isin(shared_object_ids)

            if row_filter is not None:
                if negated:
                    merged_df = merged_df[~row_filter]
                else:
                    merged_df = merged_df[row_filter]

        return merged_df

    def evaluate_setup_partial_results(self,
                                       predicate_results: typing.List[pd.DataFrame],
                                       additional_constraint_specifications: typing.List[typing.Tuple[str, typing.List[str], bool, typing.Union[str, typing.Set[str]]]]
                                       ):
        preprocessed_results = typing.cast(typing.List[pd.DataFrame], [self._preprocess_results(result, to_strings=False) for result in predicate_results])
        merged_df = None

        for i, result in enumerate(preprocessed_results):
            if merged_df is None:
                merged_df = result.rename(columns={'intervals': f'intervals_{i + 1}'})
            else:
                merge_columns = list(set(merged_df.columns) & set(result.columns) - MERGE_IGNORE_COLUMNS)
                merged_df = merged_df.merge(result, on=merge_columns, how='inner', suffixes=('', f'_{i + 1}'))

        if merged_df is None:
            raise ValueError('No results found in at-end evaluation')

        if additional_constraint_specifications is not None and len(additional_constraint_specifications) > 0:
            merged_df = self._handle_additional_constraints(merged_df, additional_constraint_specifications)

        intervals_columns = [col for col in merged_df.columns if col.startswith('intervals')]
        filter_col = merged_df[intervals_columns[0]]
        for col in intervals_columns[1:]:
            filter_col = filter_col & merged_df[col]

        return merged_df.trace_id[filter_col != 0].unique().tolist()


def _query_trace_id_to_length(connection):
    results = connection.execute('SELECT trace_id, length FROM trace_length_and_domains;').fetchall()
    return {trace_id: length for trace_id, length in results}


def _query_domain_to_trace_ids(connection):
    results = connection.execute('SELECT domain, trace_id FROM trace_length_and_domains;').fetchall()
    domain_to_trace_ids = defaultdict(list)
    for domain, trace_id in results:
        domain_to_trace_ids[domain].append(trace_id)

    return domain_to_trace_ids


def _query_trace_id_to_domain(connection):
    results = connection.execute('SELECT trace_id, domain FROM trace_length_and_domains;').fetchall()
    return {trace_id: domain for trace_id, domain in results}


def _make_defaultdict_int():
    return defaultdict(int)


def _argument_types_to_object_ids(arg_types: typing.List[str], domain: typing.Optional[str] = None) -> typing.Set[str]:
    if domain is not None:
        domain_types_to_objects = OBJECTS_BY_ROOM_AND_TYPE[domain]
        return set(sum((domain_types_to_objects.get(arg_type, []) + SPECIFIC_NAMED_OBJECTS_BY_ROOM[domain].get(arg_type, [])
                                        for arg_type in arg_types), []))

    object_ids = set()
    for domain, domain_types_to_objects in OBJECTS_BY_ROOM_AND_TYPE.items():
        object_ids.update(_argument_types_to_object_ids(arg_types, domain))

    return object_ids


DATABASE_MAX_CHILD_ARGS = 4
DATABASE_QUERY_TIMEOUT = 15


class TraceFinderASTParser(ast_parser.ASTParser):
    boolean_parser: ast_parser.ASTBooleanParser
    databse_confirmed_traces_by_preference_or_section: typing.Dict[str, typing.Union[typing.Set[str], str]]
    domain_to_trace_ids: typing.Dict[str, typing.List[str]]
    expected_keys: typing.Set[str]
    not_implemented_predicate_counts_by_preference_or_section: typing.DefaultDict[str, typing.DefaultDict[str, int]]
    predicate_data_estimator: compile_predicate_statistics_full_database.CommonSensePredicateStatisticsFullDatabase
    preferences_or_sections_with_implemented_predicates: typing.Set[str]
    setup_additional_constraints: typing.List[typing.Tuple[str, typing.List[str], bool, typing.Union[str, typing.Set[str]]]]
    setup_partial_results: typing.List[typing.Union[typing.Set[str], str, None]]
    preference_logic_evaluator: PreferenceStateMachineLogicEvaluator
    trace_names_hash: str
    traces_by_preference_or_section: typing.Dict[str, typing.Set[str]]


    def __init__(self, trace_names_hash: str = FULL_DATASET_TRACES_HASH):
        self.not_implemented_predicate_counts_by_preference_or_section = defaultdict(_make_defaultdict_int)
        self.trace_names_hash = trace_names_hash
        self._init_predicate_data_estimator()
        self.boolean_parser = BOOLEAN_PARSER
        self.preference_logic_evaluator = PreferenceStateMachineLogicEvaluator(_query_trace_id_to_length(self.predicate_data_estimator.con))
        self.domain_to_trace_ids = _query_domain_to_trace_ids(self.predicate_data_estimator.con)

    def _init_predicate_data_estimator(self):
        self.predicate_data_estimator = compile_predicate_statistics_full_database.CommonSensePredicateStatisticsFullDatabase.get_instance(
            force_trace_names_hash=self.trace_names_hash,
        )
        self.predicate_data_estimator.max_child_args = DATABASE_MAX_CHILD_ARGS
        self.query_timeout = DATABASE_QUERY_TIMEOUT

    def __getstate__(self) -> typing.Dict[str, typing.Any]:
        state = self.__dict__.copy()
        if 'predicate_data_estimator' in state:
            del state['predicate_data_estimator']
        return state

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        self.__dict__.update(state)
        self._init_predicate_data_estimator()

    def __call__(self, ast, **kwargs):
        initial_call = 'inner_call' not in kwargs or not kwargs['inner_call']
        if initial_call:
            kwargs['inner_call'] = True
            kwargs['local_context'] = {'mapping': {VARIABLES_CONTEXT_KEY: {}}}
            kwargs['global_context'] = {}
            self.expected_keys = set()
            self.traces_by_preference_or_section = {}
            self.databse_confirmed_traces_by_preference_or_section = {}
            self.preferences_or_sections_with_implemented_predicates = set()
            self.not_implemented_predicate_counts_by_preference_or_section = defaultdict(_make_defaultdict_int)
            self.setup_partial_results = []
            self.setup_additional_constraints = []

        retval = super().__call__(ast, **kwargs)

        if initial_call:
            any_setup_conditions_missed = None
            if len(self.setup_partial_results) > 0:
                # Setup
                if any(isinstance(result, str) and result == EMPTY_SET for result in self.setup_partial_results):
                    game_id = 'game'
                    if ast[1] is not None and 'game_id' in ast[1]:
                        game_id = ast[1].game_id

                    print(f'Found unsatisfiable setup conditions in {game_id}')
                    return EMPTY_SET

                any_setup_conditions_missed = any(result is None for result in self.setup_partial_results)
                filtered_setup_partial_results = [result for result in self.setup_partial_results if isinstance(result, pd.DataFrame)]
                if len(filtered_setup_partial_results) > 0:
                    setup_partial_results = self.preference_logic_evaluator.evaluate_setup_partial_results(filtered_setup_partial_results, self.setup_additional_constraints)  # type: ignore

                    if len(setup_partial_results) > 0:
                        self.databse_confirmed_traces_by_preference_or_section[ast_parser.SETUP] = setup_partial_results

            # If we missed any setup conditions, check if we can filter by remaining ones we did evaluate, since it's a conjunction
            if any_setup_conditions_missed:
                setup_traces = self.traces_by_preference_or_section.get(ast_parser.SETUP, None)
                if setup_traces is None:
                    if len(self.not_implemented_predicate_counts_by_preference_or_section[ast_parser.SETUP]) > 0:
                        print(f'Found unimplemnted setup conditions in {ast[1].game_id}')
                        setup_traces = set(self.predicate_data_estimator.all_trace_ids)

                    else:
                        raise ValueError('Setup conditions missed, but no traces found for setup')

                if ast_parser.SETUP in self.databse_confirmed_traces_by_preference_or_section:
                    setup_traces.intersection_update(self.databse_confirmed_traces_by_preference_or_section[ast_parser.SETUP])

                if len(setup_traces) == 0:
                    return EMPTY_SET

                del self.databse_confirmed_traces_by_preference_or_section[ast_parser.SETUP]

            # Handle wildcards returned by the predicate data estimator
            for key, trace_ids in self.traces_by_preference_or_section.items():
                if WILDCARD in trace_ids:
                    trace_ids.remove(WILDCARD)
                    # If the only thing we had is a wildcard, we accept all traces
                    if len(trace_ids) == 0:
                        trace_ids.update(self.predicate_data_estimator.all_trace_ids)

                    # Otherwise, we're constrained by the other predicates, so we do nothing with the wildcard

                if EMPTY_SET in trace_ids:
                    logger.info(f'Found unsatisfiable preference {key} in {ast[1].game_id}')  # type: ignore
                    return EMPTY_SET

            return self.traces_by_preference_or_section, self.expected_keys, self.databse_confirmed_traces_by_preference_or_section
        else:
            return retval

    def _current_ast_to_contexts_hook(self, ast: tatsu.ast.AST, kwargs: typing.Dict[str, typing.Any]):
        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore

        if rule == 'preference':
            kwargs['local_context']['current_preference_name'] = ast.pref_name

    def _query_predicate_data(
            self, ast: tatsu.ast.AST,
            return_trace_ids: bool = True, return_intervals: bool = False,
            check_all_predicate_args_game_objects: bool = True,
            **kwargs):

        if return_trace_ids == return_intervals:
            raise ValueError(f'Must return either trace ids or intervals, not neither or both, received {return_trace_ids} for both')

        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore
        if rule == 'super_predicate':
            ast = typing.cast(tatsu.ast.AST, ast.pred)
            rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore

        context_variables = kwargs['local_context']['mapping'][VARIABLES_CONTEXT_KEY]
        full_mapping = {k: v.var_types for k, v in context_variables.items()} if context_variables is not None else {}  # type: ignore
        relevant_variables = ast_parser.extract_variables(ast)
        relevant_mapping = {k: v for k, v in full_mapping.items() if k in relevant_variables}

        # Check for a single predicate with a mapping of only the generic `game_object`
        if check_all_predicate_args_game_objects and rule == 'predicate' and len(relevant_mapping) > 0 and all(room_and_object_types.GAME_OBJECT in var_types for var_types in relevant_mapping.values()):
            return WILDCARD

        if rule in PREDICATE_CONJUNCTION_DISJUNCTION_RULES:
            # Check that this is the type of expression we choose to support currently
            children = ast[RULE_TO_CHILD_KEY[rule]]
            if isinstance(children, list):
                children = typing.cast(typing.List[tatsu.ast.AST], children)

                for child in children:
                    child_pred_rule = child.pred.parseinfo.rule  # type: ignore
                    if child_pred_rule == 'predicate' or \
                        (child_pred_rule == 'super_predicate_not' and child.pred.not_args.pred.parseinfo.rule == 'predicate'):  # type: ignore
                        continue

                    # Found a child that is not a predicate or a negation of a predicate, so we can't handle this
                    else:
                        return

            context = {VARIABLES_CONTEXT_KEY: context_variables, SECTION_CONTEXT_KEY: kwargs[SECTION_CONTEXT_KEY]}
            logical_expr = self.boolean_parser(ast, **simplified_context_deepcopy(context))
            logical_evaluation = self.boolean_parser.evaluate_unnecessary_detailed_return_value(logical_expr)  # type: ignore

            # This expression is tautologically true, so we don't need to check it for specific objects/traces
            if logical_evaluation == 1:
                return WILDCARD

            # This expression is tautologically false, so we don't need to check it for specific objects/traces
            elif logical_evaluation == 2:
                return EMPTY_SET

            # Handle some edge cases with conjunctions of (on ...)
            if rule == 'super_predicate_and' and isinstance(children, list):
                cycle_found = False
                disconnected = False
                edge_list = []

                for p in children:
                    if p.pred.parseinfo.rule != 'predicate':  # type: ignore
                        continue

                    pred_str = ast_printer.ast_section_to_string(p, kwargs[SECTION_CONTEXT_KEY])
                    if pred_str.startswith('(on'):
                        edge_list.append(pred_str[4:-1])

                if edge_list:
                    G_directed = nx.parse_edgelist(edge_list, nodetype=str, create_using=nx.DiGraph())
                    cycle_found = len(list(nx.simple_cycles(G_directed))) > 0
                    G_undirected = G_directed.to_undirected()
                    disconnected = not nx.is_connected(G_undirected)

                if cycle_found or disconnected:
                    return EMPTY_SET

            # Handle another edge case, of some other predicates conjuncted or disjuncted with (same_type ...) or (same_object ...)
            retval = self._handle_conjunction_disjunction_same_type_object(ast, return_trace_ids=return_trace_ids, return_full_result=return_intervals, **kwargs)
            if retval is not None:
                return retval

        try:
            return self.predicate_data_estimator.filter(ast, relevant_mapping, return_trace_ids=return_trace_ids, return_full_result=return_intervals)

        except (PredicateNotImplementedException, MissingVariableException, compile_predicate_statistics_full_database.QueryTimeoutException, RuntimeError) as e:
            if isinstance(e, RuntimeError):
                args = e.args
                if isinstance(args, tuple):
                    args_str = ' '.join(str(arg) for arg in args)

                else:
                    args_str = str(args)

                if 'query interrupted' not in args_str.lower():
                    raise e

            pass

        # If we reached here, manually check a few predicates not implemented in the predicate cache for performance reasons

        done = False
        negated = False
        while not done:
            rule = ast.parseinfo.rule  # type: ignore
            if rule == 'super_predicate':
                ast = ast.pred  # type: ignore

            elif rule == 'super_predicate_not':
                negated = not negated
                ast = ast.not_args  # type: ignore
                continue

            elif rule in PREDICATE_CONJUNCTION_DISJUNCTION_RULES:
                children = ast[RULE_TO_CHILD_KEY[rule]]
                if children is not None and (not isinstance(children, list) or len(children) == 1):
                    ast = children[0]
                else:
                    done = True

            else:
                done = True

        if rule == 'predicate':
            pred = typing.cast(tatsu.ast.AST, ast.pred)
            inner_predicate_rule = pred.parseinfo.rule  # type: ignore
            # Used to handle `near` as well, but that's temporarily handled by the cache
            if inner_predicate_rule in ('predicate_equal_x_position', 'predicate_equal_z_position'):
                # If at most one argument is a furniture or room feature, we can assume it's true
                arguments = [self._term_to_terminal(arg) for arg in (pred.arg_1, pred.arg_2)]  # type: ignore
                argument_categories = [ast_parser.predicate_function_term_to_type_categories(arg, context_variables, []) for arg in arguments]
                if any(cat is None for cat in argument_categories):
                    raise ValueError(f'Argument categories not found for predicate {ast_printer.ast_section_to_string(ast, kwargs[SECTION_CONTEXT_KEY])}  with mapping {relevant_mapping}')

                argument_immobile = [any(cat in (room_and_object_types.FURNITURE, room_and_object_types.ROOM_FEATURES)
                                         for cat in categories)  # type: ignore
                                     for categories in argument_categories]

                if sum(argument_immobile) <= 1:
                    return WILDCARD if not negated else EMPTY_SET

                else:
                    logger.info(f'Found unsatisfiable predicate {ast_printer.ast_section_to_string(ast, kwargs[SECTION_CONTEXT_KEY])} with mapping {relevant_mapping}')
                    return EMPTY_SET if not negated else WILDCARD

            if inner_predicate_rule in ('predicate_same_object', 'predicate_same_type'):
                arguments = [self._term_to_terminal(arg) for arg in (pred.arg_1, pred.arg_2)]  # type: ignore

                if inner_predicate_rule == 'predicate_same_object':
                    return self._check_same_object(arguments, context_variables, negated=negated, forall=kwargs.get('forall', False))

                elif inner_predicate_rule == 'predicate_same_type':
                    return self._check_same_type(arguments, context_variables, negated=negated, forall=kwargs.get('forall', False))

        return None

    def _check_same_object(self, arguments: typing.List[str], context_variables: dict, negated: bool = False, forall: bool = False):
        argument_types = typing.cast(typing.List[typing.List[str]], [ast_parser.predicate_function_term_to_types(arg, context_variables) for arg in arguments])  # type: ignore

        valid_domains = []
        for domain in OBJECTS_BY_ROOM_AND_TYPE.keys():
            argument_objects = [_argument_types_to_object_ids(arg_types, domain) for arg_types in argument_types]

            if forall:
                if negated:
                    if not (argument_objects[0] & argument_objects[1]):
                        valid_domains.append(domain)

                else:
                    if argument_objects[0].issubset(argument_objects[1]):
                        valid_domains.append(domain)

            else:
                if negated:
                    if (argument_objects[0] - argument_objects[1]) and (argument_objects[1] - argument_objects[0]):
                        valid_domains.append(domain)

                else:
                    if argument_objects[0] & argument_objects[1]:
                        valid_domains.append(domain)

        if negated:
            valid_domains = set(OBJECTS_BY_ROOM_AND_TYPE.keys()) - set(valid_domains)

        if len(valid_domains) == 0:
            return EMPTY_SET

        elif len(valid_domains) == 3:
            return WILDCARD

        else:
            trace_ids = []
            for domain in valid_domains:
                trace_ids.extend(self.domain_to_trace_ids[domain])

            return set(trace_ids)

    def _check_same_type(self, arguments: typing.List[str], context_variables: dict, negated: bool = False, forall: bool = False):
        argument_types = typing.cast(typing.List[typing.List[str]], [ast_parser.predicate_function_term_to_types(arg, context_variables) for arg in arguments])  # type: ignore

        valid_domains = []
        for domain in OBJECTS_BY_ROOM_AND_TYPE.keys():
            argument_objects = [_argument_types_to_object_ids(arg_types, domain) for arg_types in argument_types]

            if forall:
                if negated:
                    if not (argument_objects[0] & argument_objects[1]):
                        valid_domains.append(domain)

                else:
                    if argument_objects[0].issubset(argument_objects[1]):
                        valid_domains.append(domain)

            else:
                if negated:
                    if argument_objects[0] - argument_objects[1]:
                        valid_domains.append(domain)

                else:
                    if argument_objects[0] & argument_objects[1]:
                        valid_domains.append(domain)

        if len(valid_domains) == 0:
            return EMPTY_SET

        elif len(valid_domains) == 3:
            return WILDCARD

        else:
            trace_ids = []
            for domain in valid_domains:
                trace_ids.extend(self.domain_to_trace_ids[domain])

            return set(trace_ids)

    def _handle_then(self, ast: tatsu.ast.AST, **kwargs):
        modals = [modal.seq_func for modal in ast.then_funcs]  # type: ignore
        modal_rules = [modal.parseinfo.rule for modal in modals]
        if any(modal_rule not in FULL_PREDICATE_STATE_MACHINE_SUPPORTED_MODALS for modal_rule in modal_rules):
            return

        modal_predicates = [modal[PREDICATE_IN_DATA_RULE_TO_CHILD[modal.parseinfo.rule]] for modal in modals]  # type: ignore
        modal_results = []
        modal_result_rules = []
        additional_constraints = []

        for i, (pred, modal_rule) in enumerate(zip(modal_predicates, modal_rules)):
            if modal_rule == 'while_hold':
                if pred.pred.parseinfo.rule != 'super_predicate_and':
                    return

                modal = modals[i]
                while_preds = modal['while_preds']
                if not isinstance(while_preds, list):
                    while_preds = [while_preds]

                base_pred_results = self._query_predicate_data(pred, return_trace_ids=False, return_intervals=True, **kwargs)
                if base_pred_results is None:
                    return

                if isinstance(base_pred_results, str) and base_pred_results == EMPTY_SET:
                    return EMPTY_SET

                if isinstance(base_pred_results, tuple):
                    base_pred_results, base_pred_additional_constraints = base_pred_results
                    if base_pred_additional_constraints:
                        additional_constraints.extend(base_pred_additional_constraints)

                modal_results.append(base_pred_results)
                modal_result_rules.append('hold')

                for while_pred in while_preds:
                    pred_copy = deepcopy_ast(pred, ASTCopyType.NODE)
                    pred_copy.pred.and_args.append(while_pred)
                    pred_copy_results = self._query_predicate_data(pred_copy, return_trace_ids=False, return_intervals=True, **kwargs)
                    if pred_copy_results is None:
                        return

                    if isinstance(pred_copy_results, str) and pred_copy_results == EMPTY_SET:
                        return EMPTY_SET

                    if isinstance(pred_copy_results, tuple):
                        pred_copy_results, pred_copy_additional_constraints = pred_copy_results
                        if pred_copy_additional_constraints:
                            additional_constraints.extend(pred_copy_additional_constraints)

                    modal_results.append(pred_copy_results)
                    modal_result_rules.append('hold')

                    modal_results.append(base_pred_results.copy(deep=True) if isinstance(base_pred_results, pd.DataFrame) else base_pred_results)
                    modal_result_rules.append('hold')

            else:
                try:
                    pred_results = self._query_predicate_data(pred, return_trace_ids=False, return_intervals=True, **kwargs)
                    print(modal_rule, type(pred_results) if not isinstance(pred_results, str) else pred_results)

                    if pred_results is None:
                        return

                    if isinstance(pred_results, str) and pred_results == EMPTY_SET:
                        return EMPTY_SET

                    if isinstance(pred_results, tuple):
                        pred_results, pred_additional_constraints = pred_results
                        if pred_additional_constraints:
                            additional_constraints.extend(pred_additional_constraints)

                    modal_results.append(pred_results)
                    modal_result_rules.append(modal_rule)

                except (RuntimeError, BaseException, Exception) as e:
                    error_str = f"Error evaluating predicate {ast_printer.ast_section_to_string(pred, kwargs[SECTION_CONTEXT_KEY])} with mapping {kwargs['local_context']['mapping'][VARIABLES_CONTEXT_KEY]}: {str(e)}"
                    print(error_str)
                    logger.error(error_str)
                    # try treating a timeout as something that's universally feasible
                    modal_results.append(WILDCARD)
                    modal_result_rules.append(modal_rule)


        if all(isinstance(modal_result, str) and modal_result == WILDCARD for modal_result in modal_results):
            return WILDCARD

        return set(self.preference_logic_evaluator.evaluate_then(modal_results, modal_result_rules, additional_constraints))

    def _handle_predicate_or_logical(self, ast: tatsu.ast.AST, section_or_preference_key: str, **kwargs):
        trace_ids = self._query_predicate_data(ast, return_trace_ids=True, **kwargs)

        if trace_ids is not None:
            if isinstance(trace_ids, str):
                trace_ids = {trace_ids}

            if section_or_preference_key not in self.traces_by_preference_or_section:
                self.traces_by_preference_or_section[section_or_preference_key] = set(trace_ids)
            else:
                self.traces_by_preference_or_section[section_or_preference_key].intersection_update(trace_ids)

            self.preferences_or_sections_with_implemented_predicates.add(section_or_preference_key)

            return True

        elif ast.parseinfo.rule == 'predicate':  # type: ignore
            missing_predicate = ast.pred.parseinfo.rule.replace('predicate_', '')   # type: ignore
            self.not_implemented_predicate_counts_by_preference_or_section[section_or_preference_key][missing_predicate] += 1

        return False

    def _term_to_terminal(self, term: typing.Union[tatsu.ast.AST, str]) -> str:
        while isinstance(term, tatsu.ast.AST):
            if 'term' in term:
                term = term.term  # type: ignore
                continue

            if 'terminal' in term:
                term = term.terminal   # type: ignore
                continue

            raise ValueError(f'Unexpected term {term}')

        return term

    def _check_same_type_object_or_negation(self, ast: tatsu.ast.AST, **kwargs) -> typing.Tuple[typing.Optional[str], typing.Optional[typing.List[str]], bool]:
        rule = None
        args = None
        negated = False
        done = False

        current = ast
        try:
            while not done:
                rule = current.parseinfo.rule  # type: ignore
                if rule == 'super_predicate':
                    current = current.pred  # type: ignore
                    continue

                if rule == 'super_predicate_not':
                    negated = not negated
                    current = current.not_args  # type: ignore
                    continue

                if rule == 'predicate':
                    pred = current.pred  # type: ignore
                    if pred.parseinfo.rule in ('predicate_same_type', 'predicate_same_object'):  # type: ignore
                        rule = pred.parseinfo.rule  # type: ignore
                        args = [self._term_to_terminal(arg) for arg in (pred.arg_1, pred.arg_2)]  # type: ignore
                        return rule, args, negated  # type: ignore

                break
        except:
            error_str = f'Error in _check_same_type_object_or_negation for "{ast_printer.ast_section_to_string(ast, kwargs[SECTION_CONTEXT_KEY])}"'
            print(error_str)
            logger.error(error_str)
            raise ValueError(error_str)

        return rule, args, negated  # type: ignore

    def _handle_conjunction_disjunction_same_type_object(self, ast: tatsu.ast.AST, return_full_result: bool = False, **kwargs):
        rule = typing.cast(str, ast.parseinfo.rule)   # type: ignore

        if rule not in PREDICATE_CONJUNCTION_DISJUNCTION_RULES:
            return None

        children = typing.cast(typing.List[tatsu.ast.AST], ast[RULE_TO_CHILD_KEY[rule]])

        context_variables = kwargs['local_context']['mapping'][VARIABLES_CONTEXT_KEY]

        same_type_object_indices = []
        same_type_indices_to_info = {}

        for i, child in enumerate(children):
            child_rule, args, negated = self._check_same_type_object_or_negation(child, **kwargs)
            if child_rule is not None and args is not None:
                same_type_object_indices.append(i)
                if child_rule == 'predicate_same_object':
                    child_results = self._check_same_object(args, context_variables, negated=negated, forall=kwargs.get('forall', False))

                else:  # rule ==  'same_type':
                    child_results = self._check_same_type(args, context_variables, negated=negated, forall=kwargs.get('forall', False))

                if child_results == WILDCARD and rule == 'super_predicate_or':
                    return WILDCARD

                if child_results == EMPTY_SET and rule == 'super_predicate_and':
                    return EMPTY_SET

                same_type_indices_to_info[i] = (child_rule, args, negated, child_results)

        if len(same_type_object_indices) > 0:
            new_ast = deepcopy_ast(ast, ASTCopyType.NODE)
            new_ast_children = typing.cast(typing.List[tatsu.ast.AST], new_ast[RULE_TO_CHILD_KEY[rule]])  # type: ignore
            new_ast_other_children = [new_ast_children[i] for i in range(len(new_ast_children)) if i not in same_type_object_indices]

            # if there are no non-same-type/same-object children, the result is empty
            if len(new_ast_other_children) == 0:
                return EMPTY_SET

            replace_child(new_ast,
                          RULE_TO_CHILD_KEY[rule],  # type: ignore
                          new_ast_other_children)

            kwargs['return_trace_ids'] = False
            full_result = self._query_predicate_data(new_ast, return_intervals=True, **kwargs)  # type: ignore
            if full_result is None:
                return

            if isinstance(full_result, str):
                if full_result == EMPTY_SET:
                    if rule == 'super_predicate_and':
                        return EMPTY_SET

                    else:  # at_end_pred_rule == 'super_predicate_or':
                        current_result = EMPTY_SET
                        for i, (rule, args, negated, child_results) in same_type_indices_to_info.items():
                            if child_results == WILDCARD:
                                return WILDCARD

                            if isinstance(child_results, list):
                                if isinstance(current_result, str):
                                    current_result = set(child_results)

                                else:
                                    current_result = current_result.union(child_results)

                        return current_result

                # If it's a wild card, check if any of the same type children return empty set, then return based on whether it's a disjunction or conjunction
                if full_result == WILDCARD:
                    if rule == 'super_predicate_or':
                        return WILDCARD

                    else:  # at_end_pred_rule == 'super_predicate_and':
                        current_result = WILDCARD
                        for i, (rule, args, negated, child_results) in same_type_indices_to_info.items():
                            if current_result == EMPTY_SET:
                                return EMPTY_SET

                            if isinstance(child_results, list):
                                if isinstance(current_result, str):
                                    current_result = set(child_results)

                                else:
                                    current_result = current_result.intersection(child_results)

                        return current_result

            # If we got a full result dataframe, we can filter it based on the same type children
            if isinstance(full_result, pd.DataFrame):
                downstream_constraints = []

                for i, (child_rule, args, negated, child_results) in same_type_indices_to_info.items():
                    argument_types = typing.cast(typing.List[typing.List[str]], [ast_parser.predicate_function_term_to_types(arg, context_variables) for arg in args])  # type: ignore
                    argument_objects = [_argument_types_to_object_ids(arg_types) for arg_types in argument_types]
                    shared_object_ids = argument_objects[0] & argument_objects[1]
                    row_filter = None
                    added_constraint = False

                    if child_rule == 'predicate_same_type':
                        if shared_object_ids:
                            if args[0] in full_result.columns:
                                row_filter = full_result[args[0]].isin(shared_object_ids)
                            else:
                                downstream_constraints.append((child_rule, args, negated, copy.deepcopy(context_variables)))
                                added_constraint = True

                            if args[1].startswith('?'):
                                if args[1] in full_result.columns:
                                    if row_filter is None:
                                        row_filter = full_result[args[1]].isin(shared_object_ids)
                                    else:
                                        row_filter &= full_result[args[1]].isin(shared_object_ids)
                                elif not added_constraint:
                                    downstream_constraints.append((child_rule, args, negated, copy.deepcopy(context_variables)))

                        elif not negated:
                            return EMPTY_SET

                    else:  # child_rule == 'predicate_same_object':
                        if args[0] in full_result.columns and args[1] in full_result.columns:
                            row_filter = full_result[args[0]] == full_result[args[1]]

                        # create a constraint that might get resolved later in the full `then` resolver
                        elif shared_object_ids:
                            downstream_constraints.append((child_rule, args, negated, copy.deepcopy(context_variables)))

                        elif not negated:
                            return EMPTY_SET

                    if row_filter is not None:
                        if negated:
                            full_result = full_result[~row_filter]
                        else:
                            full_result = full_result[row_filter]

                if return_full_result:
                    return full_result, downstream_constraints

                return set(full_result.trace_id.unique().tolist())

        return None

    # def _handle_at_end(self, ast: tatsu.ast.AST, **kwargs) -> typing.Optional[typing.Union[typing.Set[str], str]]:
    #     # Trying to write some code to account for a particular odd pattern our models sometimes generate:
    #     # at-ends that include (same_type, ) either inside a not or not.
    #     # Key idea: remove them, evaluate the full result without them, apply whatever limitation they provide, see what remains

    #     at_end_pred = ast.at_end_pred.pred  # type: ignore
    #     at_end_pred_rule = at_end_pred.parseinfo.rule  # type: ignore
    #     context_variables = kwargs['local_context']['mapping'][VARIABLES_CONTEXT_KEY]

    #     if at_end_pred_rule in PREDICATE_CONJUNCTION_DISJUNCTION_RULES:
    #         children = at_end_pred[RULE_TO_CHILD_KEY[at_end_pred_rule]]

    #         # Check the case of a single child
    #         if isinstance(children, tatsu.ast.AST) or len(children) == 1:
    #             if isinstance(children, list):
    #                 children = children[0]

    #             rule, args, negated = self._check_same_type_object_or_negation(children, **kwargs)
    #             if rule is not None and args is not None:
    #                 if rule == 'same_object':
    #                     result = self._check_same_object(args, context_variables, negated=negated, forall=kwargs.get('forall', False))
    #                     if isinstance(result, list):
    #                         result = set(result)

    #                     return result

    #                 else:  # rule ==  'same_type':
    #                     result = self._check_same_type(args, context_variables, negated=negated, forall=kwargs.get('forall', False))
    #                     if isinstance(result, list):
    #                         result = set(result)

    #                     return result

    #         # Check the case of multiple children -- if any fall under the above, we remove them, try to get the full result without them, and then trim appropriately
    #         else:


    #     result = self._query_predicate_data(ast.at_end_pred.pred, return_trace_ids=True, **kwargs)  # type: ignore
    #     return result  # type: ignore

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        self._current_ast_to_contexts(ast, **kwargs)
        kwargs['local_context']['mapping'] = ast_parser.update_context_variables(ast, kwargs['local_context']['mapping'])
        section_or_preference_key = kwargs['local_context']['current_preference_name'] if 'current_preference_name' in kwargs['local_context'] else (kwargs[SECTION_CONTEXT_KEY] if SECTION_CONTEXT_KEY in kwargs and kwargs[SECTION_CONTEXT_KEY] == ast_parser.SETUP else None)

        if section_or_preference_key is not None:
            self.expected_keys.add(section_or_preference_key)

        node_parsed_ignore_children = False
        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore

        if rule in ('setup_forall', 'super_predicate_forall'):
            kwargs['forall'] = True

        if rule in ('setup_game_conserved', 'setup_game_optional'):
            # if None not in self.setup_partial_results and EMPTY_SET not in self.setup_partial_results:
            if not any(x is None or (isinstance(x, str) and x == EMPTY_SET) for x in self.setup_partial_results):

                pred = typing.cast(tatsu.ast.AST, ast[PREDICATE_IN_DATA_RULE_TO_CHILD[rule]])
                result = self._query_predicate_data(pred, return_trace_ids=False, return_intervals=True, **kwargs)
                if result is not None:
                    if isinstance(result, tuple):
                        result, additional_constraints = result
                        if additional_constraints:
                            self.setup_additional_constraints.extend(additional_constraints)

                    self.setup_partial_results.append(result)  # type: ignore
                    node_parsed_ignore_children = True

        if rule == 'then':
            if section_or_preference_key is None:
                raise ValueError('Then node found without a section or preference key')

            result = self._handle_then(ast, **kwargs)
            if result is not None:
                self.databse_confirmed_traces_by_preference_or_section[section_or_preference_key] = result
                node_parsed_ignore_children = True

        if rule == 'at_end':
            if section_or_preference_key is None:
                raise ValueError('At-end node found without a section or preference key')

            result = self._query_predicate_data(ast.at_end_pred.pred, return_trace_ids=True, **kwargs)  # type: ignore
            if result is not None:
                self.databse_confirmed_traces_by_preference_or_section[section_or_preference_key] = result
                node_parsed_ignore_children = True

        if rule in PREDICATE_CONJUNCTION_DISJUNCTION_RULES:
            if section_or_preference_key is None:
                raise ValueError('Conjunction/disjunction node found without a section or preference key')

            parsed = self._handle_predicate_or_logical(ast, section_or_preference_key, **kwargs)
            if parsed:
                node_parsed_ignore_children = True

        if rule == 'predicate':  # type: ignore
            if section_or_preference_key is None:
                raise ValueError('Predicate node found without a section or preference key')

            # No need to check the return value here, since we don't need to parse children either way
            self._handle_predicate_or_logical(ast, section_or_preference_key, **kwargs)
            node_parsed_ignore_children = True

        if not node_parsed_ignore_children:
            for key in ast:
                if key != 'parseinfo':
                    child_kwargs = simplified_context_deepcopy(kwargs)
                    retval = self(ast[key], **child_kwargs)
                    self._update_contexts_from_retval(kwargs, retval)



MAX_DEPTH = 100


def recursive_check_type(obj: typing.Any, type_to_check: typing.Type, hist: typing.Optional[typing.List[typing.Union[str, int]]] = None, depth: int = 0):
    # if hist is None:
    #     hist = []

    # if isinstance(obj, type_to_check):
    #     print(hist)

    # if depth > MAX_DEPTH:
    #     return

    # if isinstance(obj, dict):
    #     for key, item in obj.items():
    #         recursive_check_type(item, type_to_check, hist + [key], depth + 1)

    # elif isinstance(obj, (list, tuple, set)):
    #     for i, item in enumerate(obj):
    #         recursive_check_type(item, type_to_check, hist + [i], depth + 1)

    # elif not isinstance(obj, (str, int, float, bool, np.ndarray)):
    #     for name, member in inspect.getmembers(obj):
    #         recursive_check_type(member, type_to_check, hist + [name], depth + 1)

    obj_iter = [('this', obj)]
    if isinstance(obj, dict):
        obj_iter = obj.items()

    elif isinstance(obj, (list, tuple, set)):
        obj_iter = enumerate(obj)

    elif not isinstance(obj, (str, int, float, bool, np.ndarray)):
        obj_iter = inspect.getmembers(obj)

    for name, item in obj_iter:
        try:
            pickler = multiprocessing.reduction.ForkingPickler(open(os.devnull, 'wb'))  # type: ignore
            pickler.dump(item)
        except TypeError as e:
            print(f'TypeError on {name} of type {type(item)}')
            traceback.print_exception(e)
            print('-=' * 40 + '-')

        except Exception as e:
            print(f'Exception on {name} of type {type(item)}')
            traceback.print_exception(e)
            print('-=' * 40 + '-')




def print_exception_detailed(exc):
    print('In print_exception_detailed')
    print(f'Exception type: {type(exc)}')
    traceback.print_exception(exc)
    if hasattr(exc, '__traceback__'):
        print('Found __traceback__')
        traceback.print_tb(exc.__traceback__, file=sys.stdout)

    print('-=' * 40 + '-')


def _load_trace(trace_id: str, base_trace_path: str = DEFAULT_BASE_TRACE_PATH):
    trace_path = os.path.join(base_trace_path, trace_id + '.json')

    if isinstance(trace_path, pathlib.Path):
        trace_path = trace_path.resolve().as_posix()

    with open(trace_path, 'r') as f:
        trace = json.load(f)
        if not isinstance(trace, list) and 'replay' in trace:
            trace = trace['replay']

    return trace


def _iterate_trace_from_id(trace_id: str, base_trace_path: str = DEFAULT_BASE_TRACE_PATH) -> typing.Iterator[typing.Tuple[dict, bool]]:
    trace = _load_trace(trace_id, base_trace_path=base_trace_path)
    for idx, event in enumerate(trace):
        yield (event, idx == len(trace) - 1)


def _iterate_trace(trace) -> typing.Iterator[typing.Tuple[dict, bool]]:
    for idx, event in enumerate(trace):
        yield (event, idx == len(trace) - 1)


def _process_id():
    identity = multiprocessing.current_process()._identity  # type: ignore
    if identity is None or len(identity) == 0:
        return 0

    return identity[0]


def _handle_single_game_single_trace_args_tuple(args_tuple) -> typing.Tuple[typing.Dict[str, typing.Dict[str, int]], typing.Dict[str, int], typing.Dict[str, int]]:
        return _handle_single_game_single_trace(*args_tuple)


def _handle_single_game_single_trace(
        trace_id: str, domain: str, sample: ASTType,
        expected_keys: typing.Set[str], ignore_keys: typing.List[str],
        base_trace_path: str = DEFAULT_BASE_TRACE_PATH, worker_tqdm: bool = False) -> typing.Tuple[typing.Dict[str, typing.Dict[str, int]], typing.Dict[str, int], typing.Dict[str, int]]:

    game_handler = GameHandler(sample,  # type: ignore
                                force_domain=domain, ignore_preference_names=ignore_keys,
                                ignore_setup=ast_parser.SETUP in ignore_keys)


    trace = _load_trace(trace_id, base_trace_path=base_trace_path)
    state_iter = _iterate_trace(trace)
    if worker_tqdm:
        state_iter = tqdm(state_iter, desc=f'Trace {trace_id} on worker {_process_id()}', total =len(trace))

    for state, is_final in state_iter:
        state = FullState.from_state_dict(state)
        score = game_handler.process(state, is_final, ignore_terminals=True)
        if score is not None:
            break

    score = game_handler.score(game_handler.scoring)
    # scores_by_trace[trace] = score

    counts_by_trace_and_key = {key: {} for key in expected_keys}
    stop_count_by_key = {key: 0 for key in expected_keys}
    total_count_by_key = {key: 0 for key in expected_keys}

    if ast_parser.SETUP in expected_keys:
        counts_by_trace_and_key[ast_parser.SETUP][trace_id] = game_handler.setup_met
        stop_count_by_key[ast_parser.SETUP] += int(game_handler.setup_met)

    for preference_name in expected_keys:
        if preference_name in game_handler.preference_satisfactions:
            n_preference_satisfcations = len(game_handler.preference_satisfactions[preference_name])
            counts_by_trace_and_key[preference_name][trace_id] = n_preference_satisfcations
            stop_count_by_key[preference_name] += int(n_preference_satisfcations > 0)
            total_count_by_key[preference_name] += n_preference_satisfcations

    return counts_by_trace_and_key, stop_count_by_key, total_count_by_key


MAX_TRACE_CACHE_SIZE = 512


class TraceGameEvaluator:
    base_trace_path: str
    chunksize: int
    force_trace_ids: typing.Optional[typing.List[str]]
    full_result_by_key: typing.Dict[typing.Union[KeyTypeAnnotation, int], typing.Dict[str, typing.Any]]
    map_elites_model_identifier: str
    max_cache_size: int
    max_traces_per_game: typing.Optional[int]
    n_days_back: int
    n_workers: int
    population: typing.Dict[typing.Union[KeyTypeAnnotation, int], ASTType]
    recompute_traces_by_population_key: bool
    result_summary_by_key: typing.Dict[typing.Union[KeyTypeAnnotation, int], int]
    save_folder: str
    save_interval: int
    save_relative_path: str
    stop_after_count: int
    temp_traces_by_population_key_path: str
    traces_by_population_key: typing.Dict[typing.Union[KeyTypeAnnotation, int], typing.Tuple[typing.List[str], typing.List[str], typing.Set[str], typing.Dict[str, typing.Dict[str, typing.List[str]]]]]
    traces_by_population_key_cache_path: str
    trace_id_to_domain: typing.Dict[str, str]
    trace_id_to_length: typing.Dict[str, int]
    trace_finder: TraceFinderASTParser
    trace_names_hash: str
    use_trace_intersection: bool
    verbose: bool
    worker_tqdm: bool

    def __init__(self, trace_names_hash: str, population: typing.Union[typing.Dict[KeyTypeAnnotation, ASTType], typing.List[ASTType]],
                 map_elites_model_identifier: str,
                 use_trace_intersection: bool = False, use_only_database_nonconfirmed_traces: bool = False,
                 stop_after_count: int = DEFAULT_STOP_AFTER_COUNT,
                 max_traces_per_game: typing.Optional[int] = None, force_trace_ids: typing.Optional[typing.List[str]] = None,
                 base_trace_path: str = DEFAULT_BASE_TRACE_PATH,
                 traces_by_population_key_cache_path: str = TRACES_BY_POPULATION_KEY_CACHE_PATH,
                 recompute_traces_by_population_key: bool = False,
                 save_folder: str = DEFAULT_MODEL_FOLDER, save_relative_path: str = DEFAULT_RELATIVE_PATH, save_interval: int = 8,
                 max_cache_size: int = MAX_TRACE_CACHE_SIZE, tqdm: bool = False,
                 n_workers: int = 1, chunksize: int = 1, worker_tqdm: bool = False,
                 maxtasksperchild: typing.Optional[int] = None,
                 n_days_back: int = DEFAULT_N_DAYS_BACK,
                 max_time_per_sample_hours: float = DEFAULT_MAX_TIME_PER_SAMPLE_HOURS,
                 verbose: bool = False):

        if isinstance(population, list):
            population = {idx: sample for idx, sample in enumerate(population)}

        self.trace_names_hash = trace_names_hash
        self.population = population
        self.map_elites_model_identifier = map_elites_model_identifier
        self.use_trace_intersection = use_trace_intersection
        self.use_only_database_nonconfirmed_traces = use_only_database_nonconfirmed_traces
        self.stop_after_count = stop_after_count
        self.max_traces_per_game = max_traces_per_game
        self.force_trace_ids = force_trace_ids
        self.base_trace_path = base_trace_path
        self.traces_by_population_key_cache_path = traces_by_population_key_cache_path
        temp_traces_by_population_key_path, ext = traces_by_population_key_cache_path.split('.', 1)
        self.temp_traces_by_population_key_path = f'{temp_traces_by_population_key_path}_{map_elites_model_identifier}.pkl'
        self.recompute_traces_by_population_key = recompute_traces_by_population_key
        self.save_folder = save_folder
        self.save_relative_path = save_relative_path
        self.save_interval = save_interval
        self.tqdm = tqdm
        self.n_workers = n_workers
        self.chunksize = chunksize
        self.worker_tqdm = worker_tqdm
        self.maxtasksperchild = maxtasksperchild
        self.n_days_back = n_days_back
        self.max_timedelta_per_sample = timedelta(hours=max_time_per_sample_hours)
        self.verbose = verbose
        self.traces_precomputed = False

        self.traces_by_population_key = {}
        if self.traces_by_population_key_cache_path is not None:
            if os.path.exists(self.traces_by_population_key_cache_path):
                trace_cache = load_data_from_path(self.traces_by_population_key_cache_path)
                if (cache_key := self._build_traces_by_population_key_cache_key()) in trace_cache:
                    action_str = 'recomputing (forced)' if self.recompute_traces_by_population_key else 'loading'
                    logger.info(f'Found traces by population key in cache for key {cache_key} -- {action_str}')
                    if not self.recompute_traces_by_population_key:
                        self.traces_by_population_key = trace_cache[cache_key]

                else:
                    logger.info(f'No traces by population key in cache for key {cache_key}')
                    print(trace_cache.keys())

        # See if there's a temp file to load
        if len(self.traces_by_population_key) == 0:
            if os.path.exists(self.temp_traces_by_population_key_path):
                with open(self.temp_traces_by_population_key_path, 'rb') as f:
                    self.traces_by_population_key = pickle.load(f)

        self.max_cache_size = max_cache_size
        self.cache = cachetools.LRUCache(maxsize=max_cache_size)
        self.trace_finder = TraceFinderASTParser(self.trace_names_hash)
        self.trace_id_to_domain = _query_trace_id_to_domain(self.trace_finder.predicate_data_estimator.con)
        self.trace_id_to_length = self.trace_finder.predicate_data_estimator.trace_id_to_length

        self.result_summary_by_key = {}
        self.full_result_by_key = {}
        self.load_partial_outputs()

    def _build_traces_by_population_key_cache_key(self):
        return f'{self.trace_names_hash}-{self.map_elites_model_identifier}'

    def __getstate__(self) -> typing.Dict[str, typing.Any]:
        state = self.__dict__.copy()

        # Avoid having these serialized since the worker processes don't need them
        if self.traces_precomputed:
            if 'trace_finder' in state:
                del state['trace_finder']

        if 'result_summary_by_key' in state:
            del state['result_summary_by_key']

        if 'full_result_by_key' in state:
            del state['full_result_by_key']

        if 'cache' in state:
            del state['cache']

        return state

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        self.__dict__.update(state)
        if 'cache' not in state:
            self.cache = cachetools.LRUCache(maxsize=self.max_cache_size)

    @cachetools.cachedmethod(operator.attrgetter('cache'))
    def _load_trace(self, trace_id: str):
        trace_path = os.path.join(self.base_trace_path, trace_id + '.json')

        if isinstance(trace_path, pathlib.Path):
            trace_path = trace_path.resolve().as_posix()

        with open(trace_path, 'r') as f:
            trace = json.load(f)
            if not isinstance(trace, list) and 'replay' in trace:
                trace = trace['replay']

        return trace

    def _iterate_trace(self, trace_id: str) -> typing.Iterator[typing.Tuple[dict, bool]]:
        trace = self._load_trace(trace_id)
        for idx, event in enumerate(trace):
            yield (event, idx == len(trace) - 1)

    def _find_key_traces(self, key: typing.Union[KeyTypeAnnotation, int]) -> typing.Tuple[typing.Union[KeyTypeAnnotation, int], typing.List[str], typing.List[str], typing.Set[str], typing.Dict[str, typing.Dict[str, typing.List[str]]]]:
        if key not in self.traces_by_population_key:
            # logger.debug(f'Finding traces for key {key} since not in cache')

            sample = self.population[key]
            retval = typing.cast(typing.Tuple[typing.Dict[str, typing.Set[str]], typing.Set[str], typing.Dict[str, typing.Dict[str, typing.List[str]]]], self.trace_finder(sample))
            if isinstance(retval, str):
                if retval == EMPTY_SET:
                    self.traces_by_population_key[key] = [], [], set(), {}

                else:
                    raise ValueError(f'Unexpected return value from trace finder: {retval}')

            else:
                traces_by_key, expected_keys, database_confirmed_traces_by_key = retval

                for sub_key, sub_key_traces in traces_by_key.items():
                    if 'domain' in sub_key_traces:
                        logger.debug(f'Found domain in traces for key {key} for sub-key {sub_key} with {len(sub_key_traces)}| {sub_key_traces}')

                all_traces = set()
                traces = []
                non_database_confirmed_trace_set = set()
                non_database_confirmed_traces = []
                initial_traces = True

                if self.use_trace_intersection:
                    for trace_set in traces_by_key.values():
                        if initial_traces:
                            all_traces.update(trace_set)
                            initial_traces = False
                        else:
                            all_traces.intersection_update(trace_set)

                    for trace_set in database_confirmed_traces_by_key.values():
                        if initial_traces:
                            all_traces.update(trace_set)
                            initial_traces = False
                        else:
                            all_traces.intersection_update(trace_set)

                    traces = list(all_traces)
                    # this doesn't really make sense when we're looking at intersections, so return the full list
                    non_database_confirmed_traces = traces

                else:
                    trace_sets_by_length = sorted(traces_by_key.values(), key=len)
                    for trace_set in trace_sets_by_length:
                        new_traces = trace_set - all_traces
                        traces.extend(new_traces)
                        all_traces.update(new_traces)

                    for pref_key, pref_key_traces in traces_by_key.items():
                        if pref_key not in database_confirmed_traces_by_key or not database_confirmed_traces_by_key[pref_key]:
                            non_database_confirmed_trace_set.update(pref_key_traces)

                    non_database_confirmed_traces = list(non_database_confirmed_trace_set)

                self.traces_by_population_key[key] = traces, non_database_confirmed_traces, expected_keys, database_confirmed_traces_by_key

        return (key, *self.traces_by_population_key[key])

    def _process_key_traces_databse_results(self, key: typing.Union[KeyTypeAnnotation, int], print_results: bool = False) -> typing.Tuple[typing.List[str], typing.List[str], typing.Dict[str, typing.Dict[str, int]], typing.Dict[str, int], typing.Dict[str, int]]:
        _, all_traces, non_database_confirmed_traces, expected_keys, database_keys_to_traces = self._find_key_traces(key)

        if print_results:
            remaining_keys = set(expected_keys) - set(database_keys_to_traces.keys())
            n_remaining_keys = len(remaining_keys)
            logger.info(f'For key {key} found {len(all_traces)} traces | {len(expected_keys)} expected keys | {len(database_keys_to_traces)} ignore keys | {n_remaining_keys} remaining keys')

        all_keys = set(expected_keys)
        all_keys.update(database_keys_to_traces.keys())

        counts_by_trace_and_key = {key: {} for key in all_keys}
        stop_count_by_key = {key: 0 for key in all_keys}
        total_count_by_key = {key: 0 for key in all_keys}

        for pref_key, trace_list in database_keys_to_traces.items():
            for trace in trace_list:
                counts_by_trace_and_key[pref_key][trace] = 1

            n_traces = len(trace_list)
            stop_count_by_key[pref_key] = min(self.stop_after_count, n_traces)
            total_count_by_key[pref_key] = n_traces

        return all_traces, non_database_confirmed_traces, counts_by_trace_and_key, stop_count_by_key, total_count_by_key

    def handle_single_game_cache_no_traces(self, key: typing.Union[KeyTypeAnnotation, int], print_results: bool = False):
        if self.force_trace_ids:
            return False, key

        all_traces, non_database_confirmed_traces, counts_by_trace_and_key, stop_count_by_key, total_count_by_key = self._process_key_traces_databse_results(key, print_results)

        # If we have no new traces, just got based on the information we already have
        if len(all_traces) == 0:
            # print('=' * 100)
            # print(f'No traces for key {key}:')
            # print(non_database_confirmed_traces)
            # print(counts_by_trace_and_key)
            # print(stop_count_by_key)
            # print(total_count_by_key)
            # print('=' * 100)

            # Verify we have any data to go by -- that it's not the case that all parts were malformed
            if len(stop_count_by_key) > 0:
                return True, key, min(stop_count_by_key.values()), counts_by_trace_and_key

            expected_keys = self._find_key_traces(key)[3]
            return True, key, 0, {expected_key: {} for expected_key in expected_keys}

        return False, key

    def _result_callback(self, single_trace_results):
        trace_counts_by_trace_and_key, trace_stop_count_by_key, trace_total_count_by_key = single_trace_results

        for pref_key in trace_counts_by_trace_and_key:
            self.current_counts_by_trace_and_key[pref_key].update(trace_counts_by_trace_and_key[pref_key])

        for pref_key in trace_stop_count_by_key:
            self.current_stop_count_by_key[pref_key] += trace_stop_count_by_key[pref_key]

        for pref_key in trace_total_count_by_key:
            self.current_total_count_by_key[pref_key] += trace_total_count_by_key[pref_key]

        # if self.current_pbar is not None:
        #     self.current_pbar.update(1)
        #     self.current_pbar.set_postfix(dict(timestamp=datetime.now().strftime('%H:%M:%S')))

        if all(stop_count >= self.stop_after_count for stop_count in self.current_stop_count_by_key.values()):
            self.current_sufficient_done = True

    def handle_single_game(self, key: typing.Union[KeyTypeAnnotation, int], print_results: bool = False,
                           pool: typing.Optional[multiprocessing.pool.Pool] = None) -> typing.Tuple[typing.Union[KeyTypeAnnotation, int], int, typing.Dict[str, typing.Dict[str, int]], typing.Optional[multiprocessing.pool.Pool]]:
        all_traces, non_database_confirmed_traces, counts_by_trace_and_key, stop_count_by_key, total_count_by_key = self._process_key_traces_databse_results(key, print_results)
        self.current_counts_by_trace_and_key = counts_by_trace_and_key
        self.current_stop_count_by_key = stop_count_by_key
        self.current_total_count_by_key = total_count_by_key
        self.current_pbar = None
        self.current_sufficient_done = False

        if self.force_trace_ids:
            traces = self.force_trace_ids

        elif self.use_only_database_nonconfirmed_traces:
            traces = non_database_confirmed_traces

        else:
            traces = all_traces

        # If we have no new traces, just got based on the information we already have
        if len(traces) == 0:
            return key, min(stop_count_by_key.values()), counts_by_trace_and_key, pool

        if self.max_traces_per_game is not None:
            traces = list(traces)[:self.max_traces_per_game]

        sorted_traces = list(sorted(traces, key=lambda trace: self.trace_id_to_length[trace]))
        sorted_trace_domains = [self.trace_id_to_domain[trace] for trace in sorted_traces]

        trace_iter = sorted_traces
        if self.tqdm: # and self.n_workers <= 1:
            trace_iter = tqdm(trace_iter, desc=f'Traces for key {key}')

        sample = self.population[key]

        _, _, _, expected_keys, database_keys_to_traces = self._find_key_traces(key)
        ignore_keys = list(database_keys_to_traces.keys())

        try:
            if self.n_workers > 1:
                n_traces = len(traces)

                if pool is None:
                    pool = multiprocessing.Pool(self.n_workers, maxtasksperchild=self.maxtasksperchild)
                    logger.info(f'Pool started for key {key}')

                # build iter for trace with args
                trace_and_args_iter = zip(
                    sorted_traces,
                    sorted_trace_domains,
                    itertools.repeat(sample, n_traces),
                    itertools.repeat(expected_keys, n_traces),
                    itertools.repeat(ignore_keys, n_traces),
                    itertools.repeat(self.base_trace_path, n_traces),
                    itertools.repeat(self.worker_tqdm, n_traces),
                )

                if self.tqdm:
                    self.current_pbar = tqdm(total=len(traces), desc=f'Traces for key {key}')

                results = [
                    pool.apply_async(  # type: ignore
                        _handle_single_game_single_trace, arg_tuple,
                        callback=self._result_callback,
                        #  error_callback=print
                        error_callback=print_exception_detailed,
                    )
                    for arg_tuple in trace_and_args_iter
                ]

                end_time = datetime.now() + self.max_timedelta_per_sample
                n_results_remaining = len(results)
                done = False
                terminate_pool = False

                while not done:
                    if self.current_sufficient_done:
                        logger.info(f'For key {key}, stopping early due to sufficient results finishing')
                        done = True
                        terminate_pool = True

                    if datetime.now() > end_time:
                        logger.warning(f'For key {key}, stopping early due to time limit')
                        done = True
                        terminate_pool = True

                    results = [result for result in results if not result.ready()]
                    current_n_results_remaining = len(results)
                    if self.current_pbar is not None and current_n_results_remaining < n_results_remaining:
                        self.current_pbar.update(n_results_remaining - current_n_results_remaining)
                        self.current_pbar.set_postfix(dict(timestamp=datetime.now().strftime('%H:%M:%S'), remaining=current_n_results_remaining, stop_counts=list(self.current_stop_count_by_key.values())))
                        n_results_remaining = current_n_results_remaining

                    if len(results) == 0:
                        logger.info(f'For key {key}, all results ready')
                        done = True

                if terminate_pool:
                    logger.info('Out of loop, about to close pool')
                    pool.close()  # type: ignore
                    logger.info('Pool closed')
                    pool.terminate()  # type: ignore
                    logger.info('Pool terminated')
                    pool.join()  # type: ignore
                    logger.info('Pool joined')
                    pool = None

                if self.current_sufficient_done:
                    logger.info('Sufficient done')
                    return key, self.stop_after_count, counts_by_trace_and_key, pool

            else:
                for trace in trace_iter:
                    domain = self.trace_id_to_domain[trace]
                    game_handler = GameHandler(sample,  # type: ignore
                                            force_domain=domain, ignore_preference_names=list(database_keys_to_traces.keys()),
                                            ignore_setup=ast_parser.SETUP in database_keys_to_traces)

                    for state, is_final in self._iterate_trace(trace):
                        state = FullState.from_state_dict(state)
                        score = game_handler.process(state, is_final, ignore_terminals=True)
                        if score is not None:
                            break

                    score = game_handler.score(game_handler.scoring)
                    # scores_by_trace[trace] = score

                    if ast_parser.SETUP in expected_keys:
                        counts_by_trace_and_key[ast_parser.SETUP][trace] = game_handler.setup_met
                        stop_count_by_key[ast_parser.SETUP] += int(game_handler.setup_met)

                    for preference_name in expected_keys:
                        if preference_name in game_handler.preference_satisfactions:
                            n_preference_satisfcations = len(game_handler.preference_satisfactions[preference_name])
                            counts_by_trace_and_key[preference_name][trace] = n_preference_satisfcations
                            stop_count_by_key[preference_name] += int(n_preference_satisfcations > 0)
                            total_count_by_key[preference_name] += n_preference_satisfcations

                    if self.verbose:
                        n_satisfactions_by_pref = " ".join(f'{k}: {len(v)}/{total_count_by_key[k]}' for k, v in game_handler.preference_satisfactions.items())
                        logger.info(f'For trace {trace} | setup met: {game_handler.setup_met} | satisfaction count: {n_satisfactions_by_pref}')

                    if all(stop_count >= self.stop_after_count for stop_count in stop_count_by_key.values()):
                        return key, self.stop_after_count, counts_by_trace_and_key, pool

            if print_results:
                for preference_name in expected_keys:
                    non_zero_count_traces = {trace: count for trace, count in counts_by_trace_and_key[preference_name].items() if count > 0 and count is not False}
                    print(f'For preference {preference_name}, {len(non_zero_count_traces)} traces have non-zero counts:')
                    for trace, count in non_zero_count_traces.items():
                        print(f'    - {trace}: {count}')
                print()
                # non_zero_score_traces = {trace: score for trace, score in scores_by_trace.items() if score != 0}
                # print(f'For key {key}, {len(non_zero_score_traces)} traces have non-zero scores, while {len(scores_by_trace) - len(non_zero_score_traces)} traces have score zero:')
                # for trace, score in non_zero_score_traces.items():
                #     print(f'    - {trace}: {score}')

            min_count = min(stop_count_by_key.values())
            return key, min_count, counts_by_trace_and_key, pool

        except Exception as e:
            logger.exception(e)
            sample_str = ast_printer.ast_to_string(sample, '\n')  # type: ignore
            logger.warning(f'The following sample produced the exception above:\n{sample_str}\n{"=" * 120}')

            if pool is not None:
                logger.info('Hit exception, about to close pool')
                pool.close()  # type: ignore
                logger.info('Pool closed')
                pool.terminate()  # type: ignore
                logger.info('Pool terminated')
                pool.join()  # type: ignore
                logger.info('Pool joined')

            return key, -1, {}, None

    def __call__(self, key: typing.Optional[KeyTypeAnnotation],
                 max_keys: typing.Optional[int] = None,
                 sort_keys_by_traces: bool = True):
        if key is not None:
            if key not in self.population:
                logger.info(f'Key {key} not found')
                return

            logger.info(f'Filtering {key}')
            self.handle_single_game(key, print_results=True)

        else:
            if self.n_workers > 1:
                key_iter, population_size = self._build_key_iter(max_keys, sort_keys_by_traces)

                logger.info(f'Traces by population keys length: {len(self.traces_by_population_key)}')

                keys_with_traces = []
                for key in key_iter:
                    if key in self.result_summary_by_key:
                        continue

                    no_traces_retval = self.handle_single_game_cache_no_traces(key)
                    if no_traces_retval[0]:
                        min_count, counts_by_trace_and_key = no_traces_retval[2:]  # type: ignore
                        self.result_summary_by_key[key] = min_count
                        self.full_result_by_key[key] = counts_by_trace_and_key

                    else:
                        keys_with_traces.append(key)

                population_size = len(keys_with_traces)
                logger.info(f'After fast processing and filtering previously processed keys, {population_size} keys have traces for full processing')

                if population_size <= 20:
                    logger.info(f'Keys with relevant traces:')
                    for key_with_traces in keys_with_traces:
                        _, all_traces, non_database_confirmed_traces, expected_keys, database_keys_to_traces = self._find_key_traces(key_with_traces)
                        remaining_keys = [pref_key for pref_key in expected_keys if pref_key not in database_keys_to_traces]
                        logger.info(f'    - {key_with_traces} | {len(all_traces)} traces | {len(expected_keys)} expected keys | {len(non_database_confirmed_traces)} traces for remaining keys | remaining keys: {remaining_keys}')


                if population_size == 0:
                    return self.result_summary_by_key, self.full_result_by_key

                # We shouldn't need it from here on out, and it will speed up creating subprocesses
                del(self.trace_finder)

                pbar = tqdm(desc='MAP-Elites samples with traces', total=population_size)
                count_since_save = 0

                sort_index = 2 if self.use_only_database_nonconfirmed_traces else 1
                keys_with_traces_and_n_traces = [(key_with_traces, len(self._find_key_traces(key_with_traces)[sort_index])) for key_with_traces in keys_with_traces]
                keys_with_traces_and_n_traces.sort(key=lambda x: x[1])
                pool = None

                for key_with_traces, _ in keys_with_traces_and_n_traces:
                    key, min_count, counts_by_trace_and_key, pool = self.handle_single_game(key_with_traces, pool=pool)
                    self.result_summary_by_key[key] = min_count
                    self.full_result_by_key[key] = counts_by_trace_and_key
                    pbar.update(1)
                    pbar.set_postfix(dict(timestamp=datetime.now().strftime('%H:%M:%S')))

                    count_since_save += 1
                    if count_since_save >= self.save_interval:
                        self.save_outputs(overwrite=True, log_message=True)
                        count_since_save = 0

            else:
                key_iter, _ = self._build_key_iter(max_keys, sort_keys_by_traces)
                for key in key_iter:
                    key, min_count, counts_by_trace_and_key, _ = self.handle_single_game(key)
                    self.result_summary_by_key[key] = min_count
                    self.full_result_by_key[key] = counts_by_trace_and_key

            print('=' * 80)
            print(self.result_summary_by_key)
            print('=' * 80)

            print('Summary of results:')
            result_counts = Counter(self.result_summary_by_key.values())
            for count in sorted(result_counts.keys()):
                print(f'    - {count}: {result_counts[count]}')

            return self.result_summary_by_key, self.full_result_by_key

    def _precompute_traces_by_population_key(self, cache_save_interval: int = 1):
        missing_keys = [key for key in self.population.keys() if key not in self.traces_by_population_key]
        if len(missing_keys) == 0:
            return

        pbar = None
        if self.tqdm:
            pbar = tqdm(total=len(missing_keys), desc=f'Initial key trace finding ({len(missing_keys)} keys)')

        last_save_count = len(self.traces_by_population_key)

        if self.n_workers > 1:
            with multiprocessing.Pool(self.n_workers, maxtasksperchild=self.maxtasksperchild) as pool:
                logger.info('Precompute traces by population key pool started')
                key_to_missing_pref_keys = {}
                for retval in pool.imap_unordered(self._find_key_traces, missing_keys, chunksize=self.chunksize):
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(dict(timestamp=datetime.now().strftime('%H:%M:%S')))

                    key, all_traces, non_database_confirmed_traces, expected_keys, ignore_keys_to_traces = retval
                    self.traces_by_population_key[key] = (all_traces, non_database_confirmed_traces, expected_keys, ignore_keys_to_traces)
                    # TODO: Temporary, remove

                    remaining_keys = set(expected_keys) - set(ignore_keys_to_traces.keys())
                    n_remaining_keys = len(remaining_keys)
                    n_ignore_keys = len(ignore_keys_to_traces)

                    min_ignore_key_traces = 100
                    ignore_keys_with_zero = []

                    for pref_key, traces in ignore_keys_to_traces.items():
                        n_traces = 0 if EMPTY_SET in traces else len(traces)
                        min_ignore_key_traces = min(min_ignore_key_traces, n_traces)
                        if n_traces == 0:
                            ignore_keys_with_zero.append(pref_key)

                    if min_ignore_key_traces == 0 or n_remaining_keys > 0:
                        print(f'For key {key} | found {len(all_traces)} traces | {len(expected_keys)} expected keys | {n_ignore_keys} ignore keys > {min_ignore_key_traces} | ignore keys with zero: {", ".join(ignore_keys_with_zero)} | {n_remaining_keys} remaining keys: {", ".join(remaining_keys)} with {len(non_database_confirmed_traces)} traces')

                    if n_remaining_keys > 0 or len(ignore_keys_with_zero) > 0:
                        key_to_missing_pref_keys[key] = dict(remaining=list(remaining_keys), ignore_with_zero=ignore_keys_with_zero)

                    if (len(self.traces_by_population_key) - last_save_count) >= cache_save_interval:
                        logger.info(f'Intermediate update (with {len(self.traces_by_population_key)} entries), saving to {self.temp_traces_by_population_key_path}')
                        save_data_to_path(self.traces_by_population_key, self.temp_traces_by_population_key_path, overwrite=True)
                        last_save_count = len(self.traces_by_population_key)

                print()
                print(key_to_missing_pref_keys)
                print()

        else:
            for key in missing_keys:
                self._find_key_traces(key)

        trace_cache = {}
        if os.path.exists(self.traces_by_population_key_cache_path):
            trace_cache = load_data_from_path(self.traces_by_population_key_cache_path)

        cache_key = self._build_traces_by_population_key_cache_key()

        logger.info(f'All done, saving trace cache for key {cache_key} to {self.traces_by_population_key_cache_path}')

        trace_cache[cache_key] = self.traces_by_population_key
        save_data_to_path(trace_cache, self.traces_by_population_key_cache_path, overwrite=True)

        logger.info('Removing temp trace file')
        os.remove(self.temp_traces_by_population_key_path)

    def _build_key_iter(self, max_keys: typing.Optional[int] = None, sort_keys_by_traces: bool = True):
        self._precompute_traces_by_population_key()
        self.traces_precomputed = True

        population_size = len(self.population)
        keys = list(self.population.keys())

        if sort_keys_by_traces:
            keys = sorted(keys, key=lambda key: len(self._find_key_traces(key)[1]), reverse=True)

        if max_keys is not None:
            keys = keys[:max_keys]
            population_size = max_keys

        if self.tqdm and self.n_workers <= 1:
            keys = tqdm(keys)

        return keys, population_size

    def save_outputs(self, overwrite: bool = False, log_message: bool = False):
        output = dict(summary=self.result_summary_by_key, full=self.full_result_by_key)
        save_data(data=output, folder=self.save_folder, name=f'trace_filter_results_{self.map_elites_model_identifier}',
                  relative_path=self.save_relative_path, overwrite=overwrite, log_message=log_message)

    def load_partial_outputs(self):
        for days_back in range(self.n_days_back):
            output_path = get_data_path(folder=self.save_folder, name=f'trace_filter_results_{self.map_elites_model_identifier}',
                                        relative_path=self.save_relative_path, delta=timedelta(days=days_back))

            if os.path.exists(output_path):
                logger.info(f'Loading partial outputs from {output_path}')
                output = load_data_from_path(output_path)
                self.result_summary_by_key = output['summary']
                self.full_result_by_key = output['full']
                return

        logger.info(f'No partial outputs found')


def build_trace_evaluator(args: argparse.Namespace) -> TraceGameEvaluator:
    multiprocessing.set_start_method(args.parallel_start_method, force=True)

    if args.copy_traces_to_tmp:
        shutil.rmtree('/tmp/participant_traces', ignore_errors=True)
        shutil.copytree(args.base_trace_path, '/tmp/participant_traces')
        logger.info('Copied traces to /tmp/participant_traces')
        args.base_trace_path = '/tmp/participant_traces'

    if args.run_from_real_games:
        logger.info('Running from real games')
        grammar = open(args.grammar_file).read()
        grammar_parser = tatsu.compile(grammar)
        population = list(cached_load_and_parse_games_from_file(args.test_file, grammar_parser, False, relative_path='.'))  # type: ignore
        model_identifier = os.path.basename(args.test_file)

    else:
        spec = None
        model_identifier = None
        if args.map_elites_model_date_id is None:
            if args.map_elites_model_name is None:
                raise ValueError('Must provide map elites model date id and name if not running from real games')

            if args.map_elites_model_name in MAP_ELITES_MODELS:
                spec = MAP_ELITES_MODELS[args.map_elites_model_name]

            else:
                model_name_year_index = args.map_elites_model_name.find(args.map_elites_run_year)
                if model_name_year_index == -1:
                    raise ValueError(f'Model name {args.map_elites_model_name} does not contain year {args.map_elites_run_year}')

                args.map_elites_model_date_id = args.map_elites_model_name[model_name_year_index:]
                args.map_elites_model_name = args.map_elites_model_name[:model_name_year_index - 1]

        if spec is not None:
            logger.info(f'Running from MAP-Elites model spec | {args.map_elites_model_name} | {spec.save_path}')
            model_identifier = spec.save_path
            model_name_year_index = model_identifier.find(args.map_elites_run_year)
            if model_name_year_index != -1:
                args.map_elites_model_date_id = model_identifier[model_name_year_index:]

            model = spec.load(relative_path=args.relative_path)

        else:
            logger.info(f'Running from MAP-Elites model | {args.map_elites_model_date_id} | {args.map_elites_model_name}')
            model_identifier = f'{args.map_elites_model_name}_{args.map_elites_model_date_id}'
            model = typing.cast(MAPElitesSampler, load_data(args.map_elites_model_date_id, args.map_elites_model_folder, args.map_elites_model_name, relative_path=args.relative_path))

        population = model.population

    trace_evaluator = TraceGameEvaluator(args.trace_names_hash,
                                            population,  # type: ignore
                                            map_elites_model_identifier=model_identifier,
                                            use_trace_intersection=args.use_trace_intersection,
                                            use_only_database_nonconfirmed_traces=args.use_only_database_nonconfirmed_traces,
                                            stop_after_count=args.stop_after_count,
                                            max_traces_per_game=args.max_traces_per_game,
                                            force_trace_ids=args.force_trace_ids,
                                            base_trace_path=args.base_trace_path,
                                            traces_by_population_key_cache_path=args.traces_by_population_key_cache_path,
                                            recompute_traces_by_population_key=args.recompute_traces_by_population_key,
                                            save_folder=args.map_elites_model_folder, save_relative_path=args.relative_path,
                                            tqdm=args.tqdm, n_workers=args.n_workers, chunksize=args.chunksize, worker_tqdm=args.worker_tqdm,
                                            save_interval=args.save_interval, maxtasksperchild=args.maxtasksperchild,
                                            n_days_back=args.n_days_back, max_time_per_sample_hours=args.max_time_per_sample_hours, verbose=args.verbose)

    return trace_evaluator


def main(args: argparse.Namespace):
    try:
        trace_evaluator = build_trace_evaluator(args)

        key = None
        if args.single_key is not None:
            key = tuple(map(int, args.single_key.replace('(', '').replace(')', '').split(',')))

        results = trace_evaluator(key, args.max_keys, not args.dont_sort_keys_by_traces)

        if results is not None:
            trace_evaluator.save_outputs(overwrite=True)

    finally:
        if args.copy_traces_to_tmp:
            shutil.rmtree('/tmp/participant_traces', ignore_errors=True)


if __name__ == '__main__':
    main(parser.parse_args())
