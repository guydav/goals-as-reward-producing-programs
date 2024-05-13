import argparse
from collections import namedtuple, defaultdict
import tatsu
import tatsu.ast
import tatsu.infos
import typing
import tqdm
import pandas as pd
import numpy as np
import os
import re

from ast_utils import load_games_from_file
from ast_parser import ASTParser
from ast_utils import update_ast
import ast_printer
from ast_to_latex_doc import extract_predicate_function_args

parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    # './dsl/problems-few-objects.pddl',
    # './dsl/problems-medium-objects.pddl',
    # './dsl/problems-many-objects.pddl',
    './dsl/interactive-beta.pddl',
)
parser.add_argument('-t', '--test-files', action='append', default=[])
parser.add_argument('-q', '--dont-tqdm', action='store_true')
DEFAULT_OUTPUT_PATH ='./data/dsl_repeated_structures_pref_body.csv'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)
parser.add_argument('-p', '--replace-predicate-names', action='store_true')
parser.add_argument('-b', '--build-function', default='build_temporal_operator_level_extractor')


PREFERENCE_BODY_STRUCTURE_STARTS = ('then', 'always', 'at-end')
TEMPORAL_OPERATOR_STRUCTURE_STARTS = ('once', 'hold', 'hold-while')
DEFAULT_HEADERS = ('structure_start', 'structure', 'count')


# TODO: rewrite this to support the latest DSL


class RepeatedStructureExtractor(ASTParser):
    def __init__(self, structure_starts, headers=DEFAULT_HEADERS):
        self.structure_starts = structure_starts
        self.headers = headers

        self.observed_structures = {start: defaultdict(int) for start in structure_starts}
        self.structure_start_rules = [start.replace('-', '_') for start in structure_starts]

        self.rule_registry = defaultdict(list)
        self.tuple_registry = defaultdict(list)
        self.regex_rules = []
        self.whitespace_re = re.compile(r'\s+')

    def _register(self, rule, replacer, tuple_rule=False):
        if tuple_rule:
            self.tuple_registry[rule].append(replacer)
        else:
            self.rule_registry[rule].append(replacer)

    def register(self, rule_or_pattern, replacer, tuple_rule=False):
        if isinstance(rule_or_pattern, re.Pattern):
            self.regex_rules.append((rule_or_pattern, replacer))

        else:
            if isinstance(rule_or_pattern, str):
                self._register(rule_or_pattern, replacer, tuple_rule)
            else:
                for rule in rule_or_pattern:
                    self._register(rule, replacer, tuple_rule)

    def to_df(self):
        rows = []
        for start, struct_to_count_dict in self.observed_structures.items():
            for structure, count in struct_to_count_dict.items():
                rows.append((start, structure, count))

        return pd.DataFrame.from_records(rows, columns=self.headers)

    def _extract_structures_from_str(self, ast_str):
        for structure_start in self.structure_starts:
            structure_start_re = re.compile(f'\\({structure_start}')
            structure_start_matches = structure_start_re.finditer(ast_str)

            for structure_start_match in structure_start_matches:
                structure_start_index = structure_start_match.start()
                n_open_parens = 0

                end_index = 0

                for end_index in range(structure_start_index, len(ast_str)):
                    if ast_str[end_index] == ')':
                        n_open_parens -= 1
                    elif ast_str[end_index] == '(':
                        n_open_parens += 1

                    if n_open_parens == 0:
                        break

                structure_end_index = end_index + 1
                structure_str = ast_str[structure_start_index:structure_end_index]
                structure_str = self.whitespace_re.sub(' ', structure_str)
                self.observed_structures[structure_start][structure_str] += 1

    def __call__(self, ast, **kwargs):
        if 'is_root' not in kwargs:
            kwargs['is_root'] = False
            is_root = True
        else:
            is_root = kwargs['is_root']

        super().__call__(ast, **kwargs)

        if is_root:
            ast_printer.reset_buffers(True)
            ast_printer.pretty_print_ast(ast)
            ast_str = ''.join(ast_printer.BUFFER)  # type: ignore
            self._extract_structures_from_str(ast_str)

        return

    def _handle_tuple(self, ast, **kwargs):
        if ast[0].startswith('(:'):
            kwargs['section'] = ast[0][2:]
        return super()._handle_tuple(ast, **kwargs)

    def _handle_ast(self, ast, **kwargs):
        if ast.parseinfo is not None:
            replacers = self.rule_registry[ast.parseinfo.rule]
            for replacer in replacers:
                replacer(ast, **kwargs)

            for (regex, replacer) in self.regex_rules:
                if regex.match(ast.parseinfo.rule):
                    replacer(ast, **kwargs)

        return super()._handle_ast(ast, **kwargs)


def _build_fake_predicate(old_parseinfo):
    fake_parseinfo = tatsu.infos.ParseInfo(old_parseinfo.tokenizer, 'predicate',
        old_parseinfo.pos, old_parseinfo.endpos, old_parseinfo.line, old_parseinfo.endline)

    return tatsu.ast.AST(pred_name='predicate', pred_args=[], parseinfo=fake_parseinfo)


def _swap_fake_predicate(ast, key):
    old_parseinfo = ast[key].parseinfo
    fake_pred = _build_fake_predicate(old_parseinfo)
    update_ast(ast, key, fake_pred)


def build_temporal_operator_level_extractor(args):
    return build_variables_and_objects_extractor(args, TEMPORAL_OPERATOR_STRUCTURE_STARTS)


def build_preference_body_level_extractor(args):
    return build_variables_and_objects_extractor(args, PREFERENCE_BODY_STRUCTURE_STARTS)


def _extract_predicate_terms(ast: tatsu.ast.AST) -> typing.List[str]:
    args = ast.pred_args

    if args is None:
        return []

    if isinstance(args, tatsu.ast.AST):
        args = [args]

    return [str(arg.term) for arg in args]


def build_variables_and_objects_extractor(args, structure_starts, replace_predicate_names=False,
    replace_non_variable_args=True, variable_replacement='?x', non_variable_replacement='?x',
    replace_comparison_numbers=True, number_replacement='0', type_replacement='object'):

    if args.replace_predicate_names:
        replace_predicate_names = args.replace_predicate_names

    extractor = RepeatedStructureExtractor(structure_starts)

    def _handle_predicate_function_args(inner_ast):
        arg_index = 1
        while f'arg_{arg_index}' in inner_ast:
            arg = inner_ast[f'arg_{arg_index}']
            if isinstance(arg, tatsu.ast.AST):
                if 'term' in arg and arg.term is not None:
                    if isinstance(arg.term, str):
                        if arg.term.startswith('?'):
                            update_ast(arg, 'term', variable_replacement)
                        else:
                            update_ast(arg, 'term', non_variable_replacement)

                    elif 'terminal' in arg.term:
                        update_ast(arg.term, 'terminal', non_variable_replacement)


            elif isinstance(arg, str):
                if arg.startswith('?'):
                    update_ast(inner_ast, f'arg_{arg_index}', variable_replacement)
                else:
                    update_ast(inner_ast, f'arg_{arg_index}', variable_replacement)
            arg_index += 1

    def handle_predicate(ast, **kwargs):
        if replace_predicate_names and 'pred_name' in ast:
            update_ast(ast, 'pred_name', 'predicate')

        _handle_predicate_function_args(ast.pred)


    extractor.register('predicate', handle_predicate)

    def handle_function_comparison(ast, **kwargs):
        inner_comp = ast.comp
        if 'equal_comp_args' in inner_comp:
            for arg_index in range(len(inner_comp.equal_comp_args)):
                inner_arg = inner_comp.equal_comp_args[arg_index].arg
                if isinstance(inner_arg, str) and inner_arg.isnumeric() and replace_comparison_numbers:
                    update_ast(inner_comp.equal_comp_args[arg_index], 'arg', number_replacement)

        else:
            for arg_key in 'arg_1', 'arg_2':
                inner_arg = inner_comp[arg_key].arg
                if isinstance(inner_arg, str) and inner_arg.isnumeric() and replace_comparison_numbers:
                    update_ast(inner_comp[arg_key], 'arg', number_replacement)

    extractor.register('function_comparison', handle_function_comparison)

    def handle_function_eval(ast, **kwargs):
        inner_func = ast.func
        _handle_predicate_function_args(inner_func)

    extractor.register('function_eval', handle_function_eval)

    def handle_exists(ast, **kwargs):
        if 'exists_vars' in ast:
            vars = ast.exists_vars.variables
            for i, var_def in enumerate(vars):
                var_names = var_def.var_names
                if isinstance(var_names, str):
                    update_ast(var_def, 'var_names', variable_replacement)

                else:
                    update_ast(var_def, 'var_names', [variable_replacement] * len(var_names))

                # TODO: consider if we want to handle (either ...) differently
                update_ast(var_def, 'var_type', type_replacement)

    extractor.register('pref_predicate_exists', handle_exists)

    def handle_forall(ast, **kwargs):
        if 'forall_vars' in ast:
            vars = ast.forall_vars.variables
            for i, var_def in enumerate(vars):
                var_names = var_def.var_names
                if isinstance(var_names, str):
                    update_ast(var_def, 'var_names', variable_replacement)

                else:
                    update_ast(var_def, 'var_names', [variable_replacement] * len(var_names))

                # TODO: consider if we want to handle (either ...) differently
                update_ast(var_def, 'var_type', type_replacement)

    extractor.register('pref_predicate_forall', handle_forall)

    return extractor

def build_oversimplified_structure_extractor(args):
    extractor = RepeatedStructureExtractor(PREFERENCE_BODY_STRUCTURE_STARTS)

    # def handle_predicate(ast, **kwargs):
    #     if 'pred_name' in ast:
    #         update_ast(ast, 'pred_name', 'predicate')

    #     if 'pred_args' in ast:
    #         # TODO: consider if I want to replace args with something more tangible
    #         update_ast(ast, 'pred_name', [])

    # extractor.register('predicate', handle_predicate)

    def handle_once(ast, **kwargs):
        if 'once_pred' in ast:
            _swap_fake_predicate(ast, 'once_pred')

    extractor.register('once', handle_once)

    def handle_hold(ast, **kwargs):
        if 'hold_pred' in ast:
            _swap_fake_predicate(ast, 'hold_pred')

    extractor.register('hold', handle_hold)

    def handle_hold_while(ast, **kwargs):
        if 'hold_pred' in ast:
            _swap_fake_predicate(ast, 'hold_pred')

        if 'while_preds' in ast:
            if isinstance(ast.while_preds, tatsu.ast.AST):
                _swap_fake_predicate(ast, 'while_preds')

            elif isinstance(ast.while_preds, list):
                for i, pred in enumerate(ast.while_preds):
                    ast.while_preds[i] = _build_fake_predicate(pred.parseinfo)

    extractor.register('while_hold', handle_hold_while)

    def handle_at_end(ast, **kwargs):
        if 'at_end_pred' in ast:
            _swap_fake_predicate(ast, 'at_end_pred')

    extractor.register('at-end', handle_at_end)

    def handle_always(ast, **kwargs):
        if 'always_pred' in ast:
            _swap_fake_predicate(ast, 'always_pred')

    extractor.register('always', handle_always)

    return extractor


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    if args.build_function in globals():
        extractor = globals()[args.build_function](args)

    else:
        raise ValueError(f'Unknown build function: {args.build_function}')

    for test_file in args.test_files:
        test_cases = load_games_from_file(test_file)

        if not args.dont_tqdm:
            test_cases = tqdm.tqdm(test_cases)

        for test_case in test_cases:
            ast = grammar_parser.parse(test_case)
            extractor(ast)

    df = extractor.to_df()

    if args.replace_predicate_names:
        args.output_path = args.output_path.replace('.csv', '_pred_names_replaced.csv')


    df.to_csv(args.output_path, index_label='Index')


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    main(args)
