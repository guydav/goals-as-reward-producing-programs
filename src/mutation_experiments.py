import argparse
from collections import defaultdict
import random
from string import ascii_lowercase
import tatsu

from ast_parser import ASTParser, ASTParentMapper
import ast_printer
from ast_utils import apply_selector_list, find_all_parents, load_asts, copy_ast, replace_child, update_ast, find_selectors_from_root


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
parser.add_argument('-t', '--test-files', action='append', default=[])
DEFAULT_NUM_GAMES = 100
parser.add_argument('-n', '--num-games', default=DEFAULT_NUM_GAMES)
# DEFAULT_OUTPUT_PATH ='./dsl_statistics.csv'
# parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)


class SamplingFailedException(Exception):
    pass


class ASTScoringPreferenceValidator(ASTParser):
    def __init__(self) -> None:
        super().__init__()
        self.pref_names = []

    def _handle_ast(self, ast, **kwargs):
        if ast.parseinfo.rule == 'preference':
            self.pref_names.append(ast.pref_name)

        elif 'count' in ast.parseinfo.rule:
            if ast.pref_name not in self.pref_names:
                update_ast(ast, 'pref_name', random.choice(self.pref_names))

        else:
            super()._handle_ast(ast, **kwargs)

    def __call__(self, ast, **kwargs):
        if 'cleared' not in kwargs:
            self.pref_names = []
            kwargs['cleared'] = True

        return super().__call__(ast, **kwargs)

class ASTVariableValidator(ASTParser):
    def _extract_and_rename_variables(self, quantifier_variables, context_variables):
        local_vars = []
        for var_def in quantifier_variables.variables:
            for i, var_name in enumerate(var_def.var_names):
                replaced = False
                while var_name in local_vars or var_name in context_variables:  # check if it already exists and we need to rename
                    var_name = f'?{random.choice(ascii_lowercase)}'
                    replaced = True

                if replaced:
                    var_def.var_names[i] = var_name

            local_vars.extend(var_def.var_names)

        return local_vars

    def _handle_ast(self, ast, **kwargs):
        local_valid_vars = self._default_kwarg(kwargs, 'local_valid_vars', [])

        # are there variables defined in this node? if so, we should add them to the list of valid variables in this context
        vars_keys = [key for key in ast.keys() if key.endswith('_vars')]
        if len(vars_keys) > 1:
            raise ValueError(f'Found multiple variables keys: {vars_keys}', ast)

        elif len(vars_keys) > 0:
            vars_key = vars_keys[0]
            lv = self._extract_and_rename_variables(ast[vars_key], local_valid_vars)
            local_valid_vars.extend(lv)

            inner_keys = [key for key in ast.keys() if key.startswith(vars_key.replace('_vars', ''))]
            inner_keys.remove(vars_key)

            if len(inner_keys) > 1:
                raise ValueError(f'Found too many inner keys: {inner_keys}', ast, ast.keys())

            inner_key = inner_keys[0]
            self(ast[inner_key], **kwargs)
            [local_valid_vars.remove(v) for v in lv]
            return

        # is this a predicate or function call? if so, we should verify that all variables are valid
        args_key = None

        if 'pred_args' in ast:
            args_key = 'pred_args'
        elif 'func_args' in ast:
            args_key = 'func_args'

        if args_key is not None:
            valid_vars_copy = local_valid_vars[:]
            used_local_vars = set()
            # if a variable already exists in this context, don't use it to replace
            for arg in ast[args_key]:
                if arg in valid_vars_copy:
                    valid_vars_copy.remove(arg)

            if isinstance(ast[args_key], str):
                if arg.startswith('?'):
                    if arg not in local_valid_vars or (arg in local_valid_vars and arg in used_local_vars):
                        if not valid_vars_copy:
                            raise SamplingFailedException(f'In replace_variables, tried to sample too many different valid variables in a single predicate')

                        new_var = random.choice(valid_vars_copy)
                        update_ast(ast, args_key, new_var)
                        valid_vars_copy.remove(new_var)
                        used_local_vars.add(new_var)

                    elif arg in local_valid_vars:
                        used_local_vars.add(arg)

            elif isinstance(ast[args_key], list):
                for i, arg in enumerate(ast[args_key]):
                    if isinstance(arg, str):
                        # check it's a variable and that it's valid in the preference or local context
                        if arg.startswith('?'):
                            if arg not in local_valid_vars or (arg in local_valid_vars and arg in used_local_vars):
                                if not valid_vars_copy:
                                    raise SamplingFailedException(f'In replace_variables, tried to sample too many different valid variables in a single predicate')

                                new_var = random.choice(valid_vars_copy)
                                ast[args_key][i] = new_var
                                valid_vars_copy.remove(new_var)
                                used_local_vars.add(new_var)

                            elif arg in local_valid_vars:
                                used_local_vars.add(arg)

                    else:
                        self(arg, **kwargs)
            else:
                raise ValueError(f'Encountered args key "{args_key}" with unexpected value type: {ast[args_key]}', ast)

        # if neither of these, do the standard ast treatment
        else:
            return super()._handle_ast(ast, **kwargs)


IGNORE_KEYS = ('parseinfo', 'mutation')


def find_mutations(ast, mutated_asts=None, should_print=True, depth=0):
    if mutated_asts is None:
        mutated_asts = list()

    if isinstance(ast, tatsu.ast.AST):
        if 'mutation' in ast:
            if should_print:  print(depth, ast)
            mutated_asts.append((depth, ast))
        for key in ast:
            if key not in IGNORE_KEYS:
                find_mutations(ast[key], mutated_asts, should_print=should_print, depth=depth+1)

    elif isinstance(ast, (tuple, list)):
        for element in ast:
            find_mutations(element, mutated_asts, should_print=should_print, depth=depth+1)

    return mutated_asts


def copy_mutation_tags(src_ast, dst_ast):
    src_mutated_nodes = find_mutations(src_ast, should_print=False)
    src_parent_mapping = ASTParentMapper()(src_ast)
    for src_mutated_node in src_mutated_nodes:
        selectors = find_selectors_from_root(src_parent_mapping, src_mutated_node, root_node=src_ast)
        dst_node = apply_selector_list(dst_ast, selectors)
        update_ast(dst_node, 'mutation', src_mutated_node['mutation'])


class ASTRegrowthSampler(ASTParser):
    def __init__(self, grammar_parser, ast_pool, n_regrowth_points=1, mutation_prob=0.2, validators=None, ignore_keys=IGNORE_KEYS) -> None:
        super().__init__()
        self.grammar_parser = grammar_parser
        self.ast_pool = ast_pool
        self.n_regrowth_points = n_regrowth_points
        self.mutation_prob = mutation_prob

        if validators is None:
            validators = list()

        self.validators = validators
        self.ignore_keys = set(ignore_keys)
        self.parent_mapper = ASTParentMapper()

    def _sample_regrowth_points(self, parent_mapping, n_regrowths):
        parent_mapping_keys = list(parent_mapping.keys())
        regrowth_points = list()

        while len(regrowth_points) < n_regrowths:
            key = random.choice(parent_mapping_keys)
            point, parent, _ = parent_mapping[key]
            # as opposed to the root tuple, which we can't assign into
            if isinstance(parent, tatsu.ast.AST):
                regrowth_points.append(point)
                for parent in find_all_parents(parent_mapping, point):
                    if isinstance(parent, tatsu.ast.AST) and parent.parseinfo in parent_mapping_keys:
                        parent_mapping_keys.remove(parent.parseinfo)

        return regrowth_points

    def _sample_ast_copy(self):
        return copy_ast(self.grammar_parser, random.choice(self.ast_pool))

    def _sample_node_by_rule(self, rule):
        result = None
        while result is None:
            ast_to_sample_from = self._sample_ast_copy()
            mapping = self.parent_mapper(ast_to_sample_from)
            rule_to_elements = defaultdict(list)
            for key, (node, _, _) in mapping.items():
                rule_to_elements[key.rule].append(node)

            if rule in rule_to_elements:
                result = random.choice(rule_to_elements[rule])

        return result

    def _validate(self, current_ast):
        for validator in self.validators:
            validator(current_ast)

        return current_ast

    def sample_regrowth(self, current_ast=None):
        if current_ast is None:
            current_ast = self._sample_ast_copy()

        current_parent_mapping = self.parent_mapper(current_ast)

        n_regrowths = self.n_regrowth_points
        if hasattr(self.n_regrowth_points, '__call__'):
            n_regrowths = self.n_regrowth_points()

        regrowth_points = self._sample_regrowth_points(current_parent_mapping, n_regrowths)

        for regrowth_point in regrowth_points:
            _, regrowth_parent, regrwoth_selector = current_parent_mapping[regrowth_point.parseinfo]
            replacement_node = self._mutate_node(regrowth_point, parent=regrowth_parent, selector=regrwoth_selector, mutation='root')
            self(replacement_node, root=True)

        return self._validate(current_ast)

    def sample_single_step(self, current_ast=None, n_mutations=0):
        if current_ast is None:
            current_ast = self._sample_ast_copy()

        current_parent_mapping = self.parent_mapper(current_ast)
        regrowth_point = self._sample_regrowth_points(current_parent_mapping, 1)[0]
        _, regrowth_parent, regrwoth_selector = current_parent_mapping[regrowth_point.parseinfo]

        relevant_keys = ['root']
        for key in regrowth_point:
            if key not in self.ignore_keys and isinstance(regrowth_point[key], (str, list, tatsu.ast.AST)):
                relevant_keys.append(key)

        key_to_replace = random.choice(relevant_keys)

        if key_to_replace == 'root':
            self._mutate_node(regrowth_point, parent=regrowth_parent, selector=regrwoth_selector, mutation=n_mutations)

        elif isinstance(regrowth_point[key_to_replace], tatsu.ast.AST):  # replace a node
            self._mutate_node(regrowth_point[key_to_replace], parent=regrowth_point, selector=[key_to_replace], mutation=n_mutations)

        else:  # replace a value
            # TODO: this is another place we could consider adding/deleting under this (if a list), rather than just replacing
            self._mutate_value(type(regrowth_point[key_to_replace]), parent=regrowth_point, selector=[key_to_replace], mutation=n_mutations)

        return self._validate(current_ast)

    def _should_mutate(self):
        return random.random() < self.mutation_prob

    def _mutate_node(self, ast, **kwargs):
        replacement_node = self._sample_node_by_rule(ast.parseinfo.rule)
        replace_child(kwargs['parent'], kwargs['selector'], replacement_node)
        mutation_type = kwargs['mutation'] if 'mutation' in kwargs else 'new'
        update_ast(replacement_node, 'mutation', mutation_type)
        return replacement_node

    def _handle_ast(self, ast, **kwargs):
        root = self._default_kwarg(kwargs, 'root', False)

        if not root:
            # do I replace this entire tree?
            if self._should_mutate():
                ast = self._mutate_node(ast, **kwargs)

        kwargs['parent'] = ast
        kwargs['root'] = False
        keys = list(ast.keys())
        for key in keys:
            if key not in self.ignore_keys:
                kwargs['selector'] = [key]
                self(ast[key], **kwargs)

    def _mutate_value(self, expected_type, **kwargs):
        parent, selector = kwargs['parent'], kwargs['selector']
        updated = False

        while not updated:
            replacement_node = self._sample_node_by_rule(parent.parseinfo.rule)
            replacement_value = replacement_node[selector[0]]
            if not isinstance(replacement_value, expected_type):
                if not replacement_value:
                    continue

                replacement_value = random.choice(replacement_value)

            if not isinstance(replacement_value, expected_type):
                # raise SamplingFailedException(f'Found an incorrectly typed value (expected a {expected_type}) even after sampling from {replacement_node[selector[0]]}')
                continue

            replace_child(parent, selector, replacement_value)
            if 'mutation' not in parent:
                mutation_type = kwargs['mutation'] if 'mutation' in kwargs else 'modified'
                update_ast(parent, 'mutation', mutation_type)

            updated = True

        return replacement_value

    def _handle_str(self, ast, **kwargs):
        # should we replace this string?
        if self._should_mutate():
            self._mutate_value(str, **kwargs)

    def _handle_list(self, ast, **kwargs):
        # should we replace the entire list?
        if self._should_mutate():
            ast = self._mutate_value(list, **kwargs)

        # TODO: if we want to consider adding/removing from a list, we would do it here
        for i, element in enumerate(ast):
            self(element, parent=kwargs['parent'], selector=kwargs['selector'] + [i])

    # TODO: do we ever want to replace values in a tuple or an int?


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    asts = load_asts(args, grammar_parser)

    # start = 152
    # for seed in range(start, start + args.num_games):
    #     validators = [ASTVariableValidator(), ASTScoringPreferenceValidator()]
    #     sampler = ASTRegrowthSampler(grammar_parser, asts, mutation_prob=0.5, validators=validators)
    #     random.seed(seed)

    #     try:
    #         # mutated_ast = sampler.sample_regrowth()
    #         mutated_ast = sampler.sample_single_step()
    #     except SamplingFailedException:
    #         continue

    #     print(f'With random seed {seed}, mutated:\r\n')
    #     # new_ast, pref_name, seq_func_index, seed = mutate_single_game(grammar_parser, asts, notebook=True, start_seed=i * args.num_games)
    #     # print(f'With random seed {seed}, mutated sequence function #{seq_func_index} in "{pref_name}":\r\n')
    #     ast_printer.reset_buffers(False)
    #     ast_printer.pretty_print_ast(mutated_ast, context=dict(html=True))
    #     print('\r\n' + '=' * 100 + '\r\n')


    first_seed = 123
    second_seed = 2

    validators = [ASTVariableValidator(), ASTScoringPreferenceValidator()]
    sampler = ASTRegrowthSampler(grammar_parser, asts, mutation_prob=0.5, validators=validators)

    random.seed(first_seed)
    src_ast = sampler.sample_single_step()

    src_ast_copy = copy_ast(grammar_parser, src_ast)
    copy_mutation_tags(src_ast, src_ast_copy)

    random.seed(second_seed)
    mutated_ast = sampler.sample_single_step(src_ast_copy, n_mutations=1)

    ast_printer.reset_buffers(False)
    ast_printer.pretty_print_ast(mutated_ast, context=dict(html=True))
    print('\r\n' + '=' * 100 + '\r\n')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
