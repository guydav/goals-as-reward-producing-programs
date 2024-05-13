import argparse
from collections import namedtuple, defaultdict
import tatsu
import tatsu.ast
import tatsu.buffering
import tqdm
import pandas as pd
import numpy as np
import os
import re

from parse_dsl import cached_load_and_parse_games_from_file
from ast_parser import ASTParser
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
DEFAULT_OUTPUT_PATH ='./data/dsl_statistics_interactive.csv'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)


class StatExtractor:
    def __init__(self, rule_or_rules, header, extract, aggregate=None):
        self.rule_or_rules = rule_or_rules
        self.header = header
        self.extract = extract
        self.aggregate = aggregate

class ASTStatisticsAggregator:
    def __init__(self):
        self.rule_registry = defaultdict(list)
        self.tuple_registry = defaultdict(list)
        self.regex_rules = []
        self.header_registry = dict()
        self.headers = ['src_file', 'game_name', 'domain_name']
        self.rows = []

    def _register(self, stat, rule, tuple_rule=False):
        if tuple_rule:
            self.tuple_registry[rule].append(stat)
        else:
            self.rule_registry[rule].append(stat)


    def register(self, stat, tuple_rule=False):
        if isinstance(stat.rule_or_rules, re.Pattern):
            self.regex_rules.append(stat)

        else:
            if isinstance(stat.rule_or_rules, str):
                self._register(stat, stat.rule_or_rules, tuple_rule)
            else:
                for rule in stat.rule_or_rules:
                    self._register(stat, rule, tuple_rule)


        self.header_registry[stat.header] = stat
        self.headers.append(stat.header)

    def to_df(self):
        return pd.DataFrame.from_records(self.rows, columns=self.headers)

    def parse(self, ast, src_file):
        row = defaultdict(list)
        row['src_file'].append(os.path.basename(src_file))
        row['game_name'].append(ast[1]["game_name"])
        row['domain_name'].append(ast[2]["domain_name"])
        ast = ast[3:]
        self._parse(ast, row)

        for header in row:
            if row[header]:
                if header in self.header_registry and self.header_registry[header].aggregate is not None:
                    row[header] = self.header_registry[header].aggregate(row[header])
                elif len(row[header]) == 1:
                    row[header] = row[header][0]

        self.rows.append(row)

    def _parse(self, ast, row, context=None):
        if context is None:
            context = dict(depth=0)

        if not ast or isinstance(ast, (str, int, tatsu.buffering.Buffer)):
            return

        elif isinstance(ast, (tuple, list)):
            if len(ast) > 0 and isinstance(ast[0], str):
                for tuple_stat in self.tuple_registry[ast[0]]:
                    result = tuple_stat.extract(ast, context)
                    if result:
                        row[tuple_stat.header].append(result)

            [self._parse(element, row, context) for element in ast]

        elif isinstance(ast, tatsu.ast.AST):
            # Look for variable definitions
            vars_keys = [key for key in ast.keys() if key.endswith('_vars')]
            if len(vars_keys) > 1:
                raise ValueError(f'Found multiple variables keys: {vars_keys}', ast)

            elif len(vars_keys) > 0:
                vars_key = vars_keys[0]
                context_vars = context['variables'] if 'variables' in context else {}

                for var_def in ast[vars_key].variables:  # type: ignore
                    var_names = var_def.var_names
                    if isinstance(var_names, str):
                        var_names = [var_names]

                    type_def = var_def.var_type.type
                    if type_def.parseinfo.rule.endswith('_type'):
                        var_types = [type_def.terminal]

                    else:  # either_x_types
                        var_types = type_def.type_names
                        if not isinstance(var_types, list):
                            var_types = [var_types]

                        var_types = [t.terminal for t in var_types]

                    for var_name in var_names:
                        context_vars[var_name] = var_types  # type: ignore

                context = context.copy()
                context['variables'] = context_vars  # type: ignore

            if ast.parseinfo is not None:
                stat_parsers = self.rule_registry[ast.parseinfo.rule]
                for stat in stat_parsers:
                    result = stat.extract(ast, context)
                    if result:
                        row[stat.header].append(result)

                for regex_stat in self.regex_rules:
                    if regex_stat.rule_or_rules.match(ast.parseinfo.rule):
                        result = regex_stat.extract(ast, context)
                        if result:
                            row[regex_stat.header].append(result)

            child_context = context.copy()
            child_context['depth'] += 1
            [self._parse(ast[child_key], row, child_context) for child_key in ast if child_key != 'parseinfo']

        else:
            print(f'Encountered AST element with unrecognized type: {ast} of type {type(ast)}')

class ASTNodeCounter(ASTParser):
    def __init__(self):
        self.count = 0

    def __call__(self, ast, **kwargs):
        if 'zero_count' in kwargs:
            self.count = 0
            del kwargs['zero_count']
        super().__call__(ast, **kwargs)
        return self.count

    def _handle_ast(self, ast, **kwargs):
        self.count += 1
        super()._handle_ast(ast, **kwargs)


def build_aggregator(args):
    agg = ASTStatisticsAggregator()

    length_of_then = StatExtractor('then', 'length_of_then', lambda ast, context: len(ast.then_funcs), lambda x: x)
    agg.register(length_of_then)

    num_preferences = StatExtractor('preference', 'num_preferences', lambda ast, context: 1, np.sum)
    agg.register(num_preferences)

    def objects_quantified(ast, context=None):
        key = 'exists_vars'
        if 'forall_vars' in ast:
            key = 'forall_vars'

        return len(ast[key]['variables'])

    num_setup_objects_quantified = StatExtractor(
        ('setup_exists', 'setup_forall'), 'setup_objects_quantified',
        objects_quantified, lambda x: x)
    agg.register(num_setup_objects_quantified)

    num_preference_objects_quantified = StatExtractor(
        ('pref_body_exists', 'pref_body_forall', 'pref_forall'), 'preference_objects_quantified',
        objects_quantified, lambda x: x)
    agg.register(num_preference_objects_quantified)

    terminal_clause_exists = StatExtractor(
        'terminal', 'terminal_exists', lambda ast, context: True, all
    )
    agg.register(terminal_clause_exists)

    def objects_referenced(ast, context=None):
        results = defaultdict(lambda: 0)

        if ast.parseinfo.rule == 'predicate':
            args = extract_predicate_function_args(ast)
            filtered_args = [arg for arg in args if isinstance(arg, str) and not arg.startswith('?')]
            for arg in filtered_args:
                    results[arg] += 1

        elif ast.parseinfo.rule == 'pref_name_and_types':
            if 'object_types' in ast:
                # single object type
                if isinstance(ast.object_types, tatsu.ast.AST):
                    results[ast.object_types.type_name.terminal] += 1

                # multiple object types
                else:
                    for type_name in [t.type_name.terminal for t in ast.object_types if 'type_name' in t]:
                        results[type_name] += 1

        else:
            key = 'exists_vars'
            if 'forall_vars' in ast:
                key = 'forall_vars'

            for quantification in ast[key]['variables']:
                var_type = quantification.var_type.type

                if var_type.parseinfo.rule.endswith('_type'):
                    results[var_type.terminal] += 1

                else:  # either_x_types
                    var_types = var_type.type_names
                    if not isinstance(var_types, list):
                        var_types = [var_types]

                    var_types = [t.terminal for t in var_types]

                    for type_name in var_types:
                        results[type_name] += 1

        return results

    def aggregate_count_dicts(count_dicts):
        results = defaultdict(lambda: 0)
        for cd in count_dicts:
            for key in cd:
                results[key] += cd[key]
        return dict(results)

    object_types_referenced = StatExtractor(
        ('setup_exists', 'setup_forall', 'setup_exists_predicate', 'setup_forall_predicate',
        'pref_body_exists', 'pref_body_forall', 'pref_forall',
        'pref_predicate_exists', 'pref_predicate_forall',
        'pref_name_and_types', 'predicate'),
        'object_types_referenced', objects_referenced, aggregate_count_dicts
    )
    agg.register(object_types_referenced)

    def extract_predicate_function_name(ast, context=None):
        if 'pred' in ast:
            ast = ast.pred

        if 'func' in ast:
            ast = ast.func

        name = ast.parseinfo.rule.replace('predicate_', '').replace('function_', '').strip()
        if name[-1].isdigit():
            name = name[:-2]

        return name

    predicates_functions_referenced = StatExtractor(
        ('predicate', 'function_eval'), 'predicates_referenced',
        lambda ast, context: {extract_predicate_function_name(ast): 1},
        aggregate_count_dicts
    )
    agg.register(predicates_functions_referenced)

    max_depth = StatExtractor('predicate', 'max_depth', lambda ast, context: context['depth'], max)
    agg.register(max_depth)

    total_ast_nodes = StatExtractor(re.compile('.*'), 'ast_nodes', lambda ast, context: 1, np.sum)
    agg.register(total_ast_nodes)

    ast_node_counter = ASTNodeCounter()

    def count_setup_nodes(ast, context=None):
        if isinstance(ast[1], tatsu.ast.AST):
            return ast_node_counter(ast[1], zero_count=True)

        return 0

    setup_nodes = StatExtractor('(:setup', 'setup_nodes', count_setup_nodes, np.sum)
    agg.register(setup_nodes, tuple_rule=True)

    def map_types_to_predicates(ast, context):
        if context is None or not isinstance(context, dict):
            raise ValueError(f'Expected a context dict, received: {context}')
        type_to_pred_counts = defaultdict(lambda: defaultdict(lambda: 0))
        inner_counts = []

        variables = context['variables'] if 'variables' in context else {}
        pred_name = extract_predicate_function_name(ast, context)
        pred_args = extract_predicate_function_args(ast)
        if isinstance(pred_args, str):
            pred_args = [pred_args]

        for arg in pred_args:
            if isinstance(arg, tatsu.ast.AST) and arg.parseinfo.rule == 'predicate':  # type: ignore
                inner_counts.append(map_types_to_predicates(arg, context))

            elif arg.startswith('?'):  # type: ignore
                if arg not in variables:
                    raise ValueError(f'Encountered undefined argument {arg} in AST: {ast}')

                for type_name in variables[arg]:
                    type_to_pred_counts[type_name][pred_name] += 1

            else:
                type_to_pred_counts[arg][pred_name] += 1

        if len(inner_counts) > 0:
            inner_counts.append(type_to_pred_counts)
            return aggregate_nested_count_dicts(inner_counts)

        return type_to_pred_counts

    def aggregate_nested_count_dicts(nested_count_dicts):
        results = defaultdict(lambda: defaultdict(lambda: 0))
        for nested_count_dict in nested_count_dicts:
            for outer_key, inner_dict in nested_count_dict.items():
                for inner_key, count in inner_dict.items():
                    results[outer_key][inner_key] += count

        return {outer_key: dict(inner_dict) for outer_key, inner_dict in results.items()}


    type_to_pred_counts = StatExtractor(('predicate', 'function_eval'), 'type_to_pred_counts', map_types_to_predicates, aggregate_nested_count_dicts)
    agg.register(type_to_pred_counts)

    return agg


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    aggregator = build_aggregator(args)

    for test_file in args.test_files:
        test_cases = cached_load_and_parse_games_from_file(test_file, grammar_parser, not args.dont_tqdm)

        for ast in test_cases:
            aggregator.parse(ast, test_file)

    df = aggregator.to_df()
    print(f'Writing statistics to {args.output_path}')
    df.to_csv(args.output_path, index_label='Index')


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    main(args)
