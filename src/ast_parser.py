from collections import namedtuple, defaultdict, OrderedDict
import copy
import itertools
import logging
import re
import typing

import boolean
import numpy as np
import tatsu
import tatsu.ast
import tatsu.buffering
import tatsu.infos


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)-8s - %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


SETUP = '(:setup'
PREFERENCES = '(:constraints'
TERMINAL = '(:terminal'
SCORING = '(:scoring'
SECTION_KEYS = (SETUP, PREFERENCES, TERMINAL, SCORING)

SECTION_CONTEXT_KEY = 'section'
VARIABLES_CONTEXT_KEY = 'variables'
QUANTIFICATIONS_CONTEXT_KEY = 'quantifications'
MODAL_CONTEXT_KEY = 'modal'
PREFERENCE_CONTEXT_KEY = 'preference'


import ast_printer
import ast_utils
import room_and_object_types


class ASTParser:
    def __call__(self, ast, **kwargs):
        # TODO: rewrite in Python 3.10-style switch-case or using a dict to map?
        if ast is None:
            return

        if isinstance(ast, str):
            return self._handle_str(ast, **kwargs)

        if isinstance(ast, (int, np.int32, np.int64)):  # type: ignore
            return self._handle_int(ast, **kwargs)

        if isinstance(ast, tuple):
            return self._handle_tuple_with_section(ast, **kwargs)

        if isinstance(ast, list):
            return self._handle_list(ast, **kwargs)

        if isinstance(ast, tatsu.ast.AST):
            return self._handle_ast(ast, **kwargs)

        if isinstance(ast, tatsu.buffering.Buffer):
            return

        raise ValueError(f'Unrecognized AST type: {type(ast)}', ast)

    def _handle_str(self, ast: str, **kwargs):
        pass

    def _handle_int(self, ast: typing.Union[int, np.int32, np.int64], **kwargs):
        pass

    def _handle_tuple_with_section(self, ast: tuple, **kwargs):
        if ast[0] in SECTION_KEYS:
            kwargs[SECTION_CONTEXT_KEY] = ast[0]

        return self._handle_tuple(ast, **kwargs)

    def _handle_tuple(self, ast: tuple, **kwargs):
        return self._handle_iterable(ast, **kwargs)

    def _handle_list(self, ast: list, **kwargs):
        return self._handle_iterable(ast, **kwargs)

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        for key in ast:
            if key != 'parseinfo':
                self(ast[key], **kwargs)

    def _handle_iterable(self, ast: typing.Iterable, **kwargs):
        return [self(item, **kwargs) for item in ast]

    def _default_kwarg(self, kwargs: typing.Dict[str, typing.Any], key: str, default_value: typing.Any, should_call: bool=False):
        if key not in kwargs or kwargs[key] is None:
            if not should_call:
                kwargs[key] = default_value
            else:
                kwargs[key] = default_value()

        return kwargs[key]

    def _update_contexts_from_retval(self, kwargs: typing.Dict[str, typing.Any], retval: typing.Any):
        if retval is not None and isinstance(retval, tuple) and len(retval) == 2:
            global_context_update, local_context_update = retval
            if global_context_update is not None:
                kwargs['global_context'].update(global_context_update)
            if local_context_update is not None:
                kwargs['local_context'].update(local_context_update)

    def _current_ast_to_contexts(self, ast: tatsu.ast.AST, **kwargs):
        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore

        if rule == 'preference':
            if 'preference_names' not in kwargs['global_context']:
                kwargs['global_context']['preference_names'] = defaultdict(int)

            kwargs['global_context']['preference_names'][ast.pref_name] += 1
            kwargs['local_context']['current_preference_name'] = ast.pref_name

        elif rule == 'variable_list':
            if isinstance(ast.variables, tatsu.ast.AST):
                self._update_variable_type_def_kwargs(ast.variables, kwargs)

            else:
                for var_def in ast.variables:  # type: ignore
                    self._update_variable_type_def_kwargs(var_def, kwargs)

        elif rule.endswith('variable_type_def'):
            self._update_variable_type_def_kwargs(ast, kwargs)

        self._current_ast_to_contexts_hook(ast, kwargs)

    def _current_ast_to_contexts_hook(self, ast: tatsu.ast.AST, kwargs: typing.Dict[str, typing.Any]):
        pass

    def _variable_type_def_rule_to_context_key(self, rule: str) -> str:
        if rule.startswith('variable'):
            return VARIABLES_CONTEXT_KEY
        else:
            return f'{rule.split("_")[0]}_{VARIABLES_CONTEXT_KEY}'

    def _update_variable_type_def_kwargs(self, ast, kwargs):
        key = self._variable_type_def_rule_to_context_key(ast.parseinfo.rule)

        if key not in kwargs['local_context']:
            kwargs['local_context'][key] = dict()

        update = self._extract_variable_def_variables(ast)
        update.update(kwargs['local_context'][key])
        kwargs['local_context'][key] = update

    def _extract_variable_def_variables(self, ast):
        var_names = ast.var_names
        if isinstance(var_names, str):
            var_names = [var_names]

        return {v[1:]: ast.parseinfo.pos for v in var_names}


ContextDict = typing.Dict[str, typing.Any]


class ASTNodeInfo(typing.NamedTuple):
    ast: tatsu.ast.AST
    parent: tatsu.ast.AST
    selector: typing.List[typing.Union[str, int]]
    depth: int
    section: typing.Optional[str]
    global_context: ContextDict
    local_context: ContextDict


ASTParentMapping = typing.Dict[tatsu.infos.ParseInfo, ASTNodeInfo]

VARIABLE_OWNER_CONTEXT_KEY_PREFIX = 'owner'
LOCAL_CONTEXT_PROPAGATING_RULES = set(['variable_type_def', 'color_variable_type_def', 'orientation_variable_type_def', 'side_variable_type_def', 'variable_list'])


class ASTParentMapper(ASTParser):
    def __init__(self, root_node='root', local_context_propagating_rules: typing.Set[str] = LOCAL_CONTEXT_PROPAGATING_RULES):
        self.root_node = root_node
        self.local_context_propagating_rules = local_context_propagating_rules
        self.parent_mapping = {}

    def __call__(self, ast, **kwargs):
        self._default_kwarg(kwargs, 'parent', ast)
        self._default_kwarg(kwargs, 'selector', [])
        self._default_kwarg(kwargs, 'depth', 0)
        self._default_kwarg(kwargs, SECTION_CONTEXT_KEY, None)
        self._default_kwarg(kwargs, 'global_context', dict())
        self._default_kwarg(kwargs, 'local_context', dict())
        kwargs['local_context'] = ast_utils.simplified_context_deepcopy(kwargs['local_context'])
        retval = super().__call__(ast, **kwargs)
        self._update_contexts_from_retval(kwargs, retval)
        return retval

    def _handle_tuple(self, ast: tuple, **kwargs):
        if isinstance(ast[0], str) and ast[0].startswith('(:'):
            kwargs[SECTION_CONTEXT_KEY] = ast[0]

        return self._handle_iterable(ast, **kwargs)

    def _handle_iterable(self, ast, **kwargs):
        base_selector = kwargs['selector']
        current_depth = kwargs['depth']
        child_propagated_context = False

        for i, element in enumerate(ast):
            kwargs['selector'] = base_selector + [i]
            kwargs['depth'] = current_depth + 1
            retval = self(element, **kwargs)
            if retval is not None and retval[1] is not None:
                child_propagated_context = True
            self._update_contexts_from_retval(kwargs, retval)

        if child_propagated_context:
            return kwargs['global_context'], kwargs['local_context']

        return kwargs['global_context'], None

    def _build_mapping_value(self, ast, **kwargs):
        return ASTNodeInfo(ast, kwargs['parent'], kwargs['selector'], kwargs['depth'], kwargs[SECTION_CONTEXT_KEY],
            ast_utils.simplified_context_deepcopy(kwargs['global_context']),
            ast_utils.simplified_context_deepcopy(kwargs['local_context']))

    def _parse_current_node(self, ast, **kwargs):
        pass

    def _should_rehandle_current_node(self, ast, **kwargs) -> typing.Tuple[bool, typing.Optional[typing.Tuple[typing.Optional[typing.Dict[str, typing.Any]], typing.Optional[typing.Dict[str, typing.Any]]]]]:
        return False, None

    def _handle_ast(self, ast, **kwargs):
        self._current_ast_to_contexts(ast, **kwargs)

        self._parse_current_node(ast, **kwargs)

        current_depth = kwargs['depth']
        for key in ast:
            if key != 'parseinfo':
                child_kwargs = kwargs.copy()

                child_kwargs['parent'] = ast
                child_kwargs['selector'] = [key]
                child_kwargs['depth'] = current_depth + 1
                retval = self(ast[key], **child_kwargs)
                # check if there's a local context update, and if there is, if we should change which node owns the variable
                if retval is not None and isinstance(retval, tuple) and ast.parseinfo.rule not in self.local_context_propagating_rules:  # type: ignore
                    local_context_update = retval[1]

                    if local_context_update is not None and local_context_update:
                        for key in list(local_context_update.keys()):
                            if key.endswith(VARIABLES_CONTEXT_KEY) and not key.startswith(VARIABLE_OWNER_CONTEXT_KEY_PREFIX):
                                owner_key = f'{VARIABLE_OWNER_CONTEXT_KEY_PREFIX}_{key}'
                                if owner_key not in local_context_update:
                                    local_context_update[owner_key] = dict()

                                for var_name in local_context_update[key]:
                                    if var_name not in local_context_update[owner_key]:
                                        local_context_update[owner_key][var_name] = ast.parseinfo.pos  # type: ignore

                self._update_contexts_from_retval(kwargs, retval)

        should_rehandle, kwarg_updates = self._should_rehandle_current_node(ast, **kwargs)
        if should_rehandle:
            self._update_contexts_from_retval(kwargs, kwarg_updates)
            self._handle_ast(ast, **kwargs)

        self._add_ast_to_mapping(ast, **kwargs)

        if self.local_context_propagating_rules and ast.parseinfo.rule in self.local_context_propagating_rules:  # type: ignore
            return kwargs['global_context'], kwargs['local_context']

        return kwargs['global_context'], None

    def _ast_key(self, ast):
        return ast.parseinfo._replace(alerts=None)

    def _add_ast_to_mapping(self, ast, **kwargs):
        self.parent_mapping[self._ast_key(ast)] = self._build_mapping_value(ast, **kwargs)  # type: ignore

    def get_parent_info(self, ast):
        return self.parent_mapping[self._ast_key(ast)]


DEFAULT_PARSEINFO_COMPARISON_INDICES = (1, 2, 3)


class ASTParseinfoSearcher(ASTParser):
    def __init__(self, comparison_indices: typing.Sequence[int] = DEFAULT_PARSEINFO_COMPARISON_INDICES):
        super().__init__()
        self.comparison_indices = comparison_indices

    def __call__(self, ast, **kwargs) -> typing.Optional[tatsu.ast.AST]:
        if 'parseinfo' not in kwargs:
            raise ValueError('parseinfo must be passed as a keyword argument')

        return super().__call__(ast, **kwargs)  # type: ignore

    def _handle_iterable(self, ast: typing.Iterable, **kwargs):
        for item in ast:
            result = self(item, **kwargs)
            if result is not None:
                return result

        return None

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        if all(ast.parseinfo[i] == kwargs['parseinfo'][i] for i in self.comparison_indices):  # type: ignore
            return ast

        for key in ast:
            if key != 'parseinfo':
                result = self(ast[key], **kwargs)
                if result is not None:
                    return result

        return None


class ASTDepthParser(ASTParser):
    def __call__(self, ast, **kwargs):
        self._default_kwarg(kwargs, 'depth', 0)
        return super().__call__(ast, **kwargs)

    def _handle_iterable(self, ast: typing.Iterable, **kwargs):
        child_values = [self(item, depth=kwargs['depth'] + 1) for item in ast]
        child_values = [cv for cv in child_values if cv is not None]
        if not child_values:
            return kwargs['depth']
        return max(child_values)

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        child_values = [self(ast[key], depth=kwargs['depth'] + 1) for key in ast if key != 'parseinfo']
        child_values = [cv for cv in child_values if cv is not None]
        if not child_values:
            return kwargs['depth']
        return max(child_values)

    def _handle_str(self, ast: str, **kwargs):
        return kwargs['depth']

    def _handle_int(self, ast: typing.Union[int, np.int32, np.int64], **kwargs):
        return kwargs['depth']


class ASTPredicateFunctionCounter(ASTParser):
    def __init__(self):
        self.counts = defaultdict(int)

    def _handle_ast(self, ast, **kwargs):
        rule = ast.parseinfo.rule  # type: ignore
        if rule == 'function_eval':
            inner_rule = ast.func.parseinfo.rule  # type: ignore
            self.counts[inner_rule.replace('function_', '')] += 1

        elif rule == 'predicate':
            inner_rule = ast.pred.parseinfo.rule  # type: ignore
            self.counts[inner_rule.replace('predicate_', '')] += 1

        return super()._handle_ast(ast, **kwargs)



VariableDefinition = namedtuple('VariableDefinition', ('var_names', 'var_types', 'parseinfo'))


def extract_variables_from_ast(ast: tatsu.ast.AST, vars_key: str, context_vars: typing.Dict[str, typing.Union[VariableDefinition, typing.List[VariableDefinition]]]) -> None:
    variables = ast[vars_key].variables  # type: ignore

    if variables is None:
        logger.warning(f'No variables found in {ast}')
        return

    if isinstance(variables, tatsu.ast.AST):
        variables = [variables]

    for var_def in variables:  # type: ignore
        var_names = var_def.var_names
        if isinstance(var_names, str):
            var_names = [var_names]

        var_type = var_def.var_type.type  # type: ignore

        if isinstance(var_type, tatsu.ast.AST):
            var_type = _extract_variable_type_as_list(var_type)

        if isinstance(var_type, str):
            var_type = [var_type]

        variable_definition = VariableDefinition(var_names, var_type, var_def.parseinfo)

        for var_name in var_names:  # type: ignore
            if var_name in context_vars:
                if isinstance(context_vars[var_name], list):
                    context_vars[var_name].append(variable_definition)  # type: ignore
                else:
                    context_vars[var_name] = [context_vars[var_name], variable_definition]  # type: ignore

            context_vars[var_name] = variable_definition


def update_context_variables(ast: tatsu.ast.AST, context: typing.Dict[str, typing.Any]) -> dict:
    vars_keys = [key for key in ast.keys() if key.endswith('_vars')]
    if len(vars_keys) > 1:
        raise ValueError(f'Found multiple variables keys: {vars_keys}', ast)

    elif len(vars_keys) > 0:
        vars_key = vars_keys[0]
        context_vars = typing.cast(dict, context.get(VARIABLES_CONTEXT_KEY, {}))
        extract_variables_from_ast(ast, vars_key, context_vars)
        context = context.copy()
        context[VARIABLES_CONTEXT_KEY] = context_vars  # type: ignore
        if QUANTIFICATIONS_CONTEXT_KEY not in context:
            context[QUANTIFICATIONS_CONTEXT_KEY] = 0

        context[QUANTIFICATIONS_CONTEXT_KEY] += 1

    return context


def predicate_function_term_to_types(term_or_terms: typing.Union[str, typing.List[str]],
    context_variables: typing.Dict[str, typing.Union[VariableDefinition, typing.List[VariableDefinition]]],
    ) -> typing.Optional[typing.List[str]]:

    if isinstance(term_or_terms, str):
        term_or_terms = [term_or_terms]

    term_type_list = []

    for term in term_or_terms:
        if term.startswith('?'):
            if term in context_variables:
                var_def = context_variables[term]
                if isinstance(var_def, list):
                    var_types = var_def[0].var_types
                else:
                    var_types = var_def.var_types

                if isinstance(var_types, list):
                    term_type_list.extend(var_types)
                else:
                    term_type_list.append(var_types)
        else:
            term_type_list.append(term)

    if any(not isinstance(term_type, str) for term_type in term_type_list):
        print(f'Non-string type found in {term_type_list}')

    term_types = dict()  # OrderedDict()
    for type in term_type_list:
        if type not in term_types:
            term_types[type] = 0

    return list(term_types.keys())


def predicate_function_term_to_type_categories(term_or_terms: typing.Union[str, typing.List[str]],
    context_variables: typing.Dict[str, typing.Union[VariableDefinition, typing.List[VariableDefinition]]],
    known_missing_types: typing.Iterable[str]) -> typing.Optional[typing.List[str]]:

    term_types = predicate_function_term_to_types(term_or_terms, context_variables)

    if not term_types:
        return None

    term_categories = dict()  # OrderedDict()
    for term_type in term_types:
        if term_type not in room_and_object_types.TYPES_TO_CATEGORIES:
            if term_type not in known_missing_types and not term_type.isnumeric():
                continue
                # print(f'Unknown type {term_type_list} not in the types to categories map')
        else:
            term_category = room_and_object_types.TYPES_TO_CATEGORIES[term_type]
            if term_category not in term_categories:
                term_categories[term_category] = 0

    return list(term_categories.keys())


DEFAULT_MAX_TAUTOLOGY_EVAL_LENGTH = 16
OPPOSITEֹֹֹ_CONTRADICTS_PREDICATES = ['above', 'in', 'on']
OPPOSITE_REDUNDANT_PREDICATE = ['adjacent', 'opposite', 'touch', 'same_color', 'same_object', 'same_type']
DOUBLE_UNDERSCORE_NUMBER_PATTERN = re.compile(r'__-?\d+')
NUMBER_SUB = '__NUMBER'


class ASTBooleanParser(ASTParser):
    # node_to_symbol_mapping: typing.Dict[str, boolean.Symbol]
    algebra: boolean.BooleanAlgebra
    next_arithmetic_operator_as_and: bool
    str_to_expression_mapping: typing.Dict[str, boolean.Expression]
    # valid_symbol_names: typing.List[str]
    whitespace_pattern: re.Pattern
    def __init__(self, max_tautology_eval_length: int = DEFAULT_MAX_TAUTOLOGY_EVAL_LENGTH,
                 opposite_contradicts_predicates: typing.List[str] = OPPOSITEֹֹֹ_CONTRADICTS_PREDICATES,
                 opposite_redundant_predicates: typing.List[str] = OPPOSITE_REDUNDANT_PREDICATE):
        self.max_tautology_eval_length = max_tautology_eval_length
        self.opposite_contrdicts_predicates = opposite_contradicts_predicates
        self.opposite_redundant_predicates = opposite_redundant_predicates
        self.opposite_unnecesary_predicates = opposite_contradicts_predicates + opposite_redundant_predicates
        # self.next_arithmetic_operator_as_and = True

        self.algebra = boolean.BooleanAlgebra()
        self.true = self.algebra.parse('TRUE')
        self.false = self.algebra.parse('FALSE')
        self.whitespace_pattern = re.compile(r'\s+')
        self.variable_pattern = re.compile(r'\?[\w\d]+')
        self.number_pattern = re.compile(r'-?\d*\.?\d+')
        # self.valid_symbol_names = list(''.join((l, d)) for d, l in itertools.product([''] + [str(x) for x in range(5)], string.ascii_lowercase,))
        self.game_start()

    def game_start(self):
        self.str_to_expression_mapping = {}
        self.next_symbol_name_index = 0
        # self.next_arithmetic_operator_as_and = True

    def _all_equal(self, iterable: typing.Iterable):
        # Returns True if all the elements are equal to each other -- from Python itertools recipes
        g = itertools.groupby(iterable)
        return next(g, True) and not next(g, False)

    def evaluate_unnecessary_detailed_return_value(self, expr: boolean.Expression) -> int:
        if not isinstance(expr, (boolean.AND, boolean.OR)):
            return 0

        simplified = expr.simplify()
        if simplified == self.true:
            return 1

        if simplified == self.false:
            return 2

        for arg in expr.args:
            if isinstance(arg, boolean.Symbol):
                if arg.obj[0] == '(' and arg.obj[-1] == ')':
                    arg_text = arg.obj[1:-1]
                    arg_components = arg_text.split('__')
                    predicate = arg_components[0]
                    if len(arg_components) == 3 and predicate in self.opposite_unnecesary_predicates:
                        opposite_predicate_symbol = boolean.Symbol(f'({predicate}__{arg_components[2]}__{arg_components[1]})')

                        if opposite_predicate_symbol in expr.args:
                            if isinstance(expr, boolean.AND) and predicate in self.opposite_contrdicts_predicates:
                                return 2

                            return 1

        return 0

    def evaluate_unnecessary(self, expr: boolean.Expression) -> bool:
        return self.evaluate_unnecessary_detailed_return_value(expr) > 0

    def evaluate_tautology(self, expr: boolean.Expression) -> bool:
        symbols = list(expr.symbols)

        if len(symbols) > self.max_tautology_eval_length:
            # print(f'Not evaluating tautology for expression with {len(symbols)} symbols')
            return False

        # initial_value = None
        # found_both_values = False

        # # for value_assignments in itertools.product([self.true, self.false], repeat=len(symbols)):
        # #     value = expr.subs({s: v for s, v in zip(symbols, value_assignments)}, simplify=True)
        # #     if initial_value is None:
        # #         initial_value = value
        # #     elif initial_value != value:
        # #         found_both_values = True
        # #         break

        # # return not found_both_values

        # possible_values = [
        #     expr.subs({s: v for s, v in zip(symbols, value_assignments)}, simplify=True)
        #     for value_assignments in itertools.product([self.true, self.false], repeat=len(symbols))
        # ]
        # return self._all_equal(possible_values)
        return self._all_equal(expr.subs({s: v for s, v in zip(symbols, value_assignments)}, simplify=True)
            for value_assignments in itertools.product([self.true, self.false], repeat=len(symbols)))

    def evaluate_redundancy(self, expr: boolean.Expression) -> bool:
        simplified = expr.simplify()
        redundancy_detected = (len(str(simplified)) < len(str(expr))) and (len(simplified.args) <= len(expr.args))
        return redundancy_detected

    def __call__(self, ast: tatsu.ast.AST, **kwargs) -> typing.Union[boolean.Expression, typing.List[boolean.Expression]]:
        if SECTION_CONTEXT_KEY not in kwargs:
            raise ValueError(f'Context key {SECTION_CONTEXT_KEY} not found in kwargs')
        return super().__call__(ast, **kwargs)  # type: ignore

    def _key_to_symbol_str(self, key: str):
        return key.replace(') )', '))').replace('?', '').replace(' ', '__')

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs) -> boolean.Expression:
        rule = typing.cast(str, ast.parseinfo.rule)  # type: ignore
        key = self._ast_to_string_key(ast, **kwargs)

        if VARIABLES_CONTEXT_KEY in kwargs:
            variables_in_key = set(self.variable_pattern.findall(key))

            for var in variables_in_key:
                if var in kwargs[VARIABLES_CONTEXT_KEY]:
                    var_def = kwargs[VARIABLES_CONTEXT_KEY][var]
                    if isinstance(var_def, list):
                        var_def = var_def[0]

                    var_types = var_def.var_types
                    var_types_str = '|'.join(var_types)
                    key = key.replace(var, f'{var}-{var_types_str}')

        if key in self.str_to_expression_mapping:
            return self.str_to_expression_mapping[key]

        expr = None

        if rule.endswith('_and'):
            arg_mappings = self(ast.and_args, **kwargs)  # type: ignore
            if isinstance(arg_mappings, list) and len(arg_mappings) == 1:
                arg_mappings = arg_mappings[0]

            if isinstance(arg_mappings, boolean.Expression):
                expr = arg_mappings

            else:
                expr = boolean.AND(*arg_mappings)

        elif rule.endswith('_or'):
            arg_mappings = typing.cast(list, self(ast.or_args, **kwargs))  # type: ignore
            if isinstance(arg_mappings, list) and len(arg_mappings) == 1:
                arg_mappings = arg_mappings[0]

            if isinstance(arg_mappings, boolean.Expression):
                expr = arg_mappings

            else:
                expr = boolean.OR(*arg_mappings)

        elif rule.endswith('_not'):
            arg_mapping = self(ast.not_args, **kwargs)  # type: ignore
            expr = boolean.NOT(arg_mapping)

        # elif rule == 'setup':
        #     expr = self(ast.setup, **kwargs)  # type: ignore

        # elif rule == 'setup_statement':
        #     expr = self(ast.statement, **kwargs)  # type: ignore

        elif rule.endswith('_exists'):
            var_dict = kwargs[VARIABLES_CONTEXT_KEY] if VARIABLES_CONTEXT_KEY in kwargs else {}
            extract_variables_from_ast(ast, 'exists_vars', var_dict)
            kwargs = kwargs.copy()
            kwargs[VARIABLES_CONTEXT_KEY] = var_dict

            expr = self(ast.exists_args, **kwargs)  # type: ignore
            expr = boolean.Symbol(f'exists_{str(expr).lower()}')

        elif rule.endswith('_forall'):
            var_dict = kwargs[VARIABLES_CONTEXT_KEY] if VARIABLES_CONTEXT_KEY in kwargs else {}
            extract_variables_from_ast(ast, 'forall_vars', var_dict)
            kwargs = kwargs.copy()
            kwargs[VARIABLES_CONTEXT_KEY] = var_dict

            expr = self(ast.forall_args, **kwargs)  # type: ignore
            expr = boolean.Symbol(f'forall_{str(expr).lower()}')

        elif rule == 'super_predicate':
            expr = self(ast.pred, **kwargs)  # type: ignore

        elif rule == 'predicate':
            expr = boolean.Symbol(self._key_to_symbol_str(key))

        elif rule in ('two_arg_comparison', 'multiple_args_equal_comparison', 'terminal_comp', 'scoring_comp', 'scoring_equals_comp'):
            if rule == 'terminal_comp':
                ast = typing.cast(tatsu.ast.AST, ast.comp)

            if 'comp_op' in ast:
                op = ast.comp_op
            elif 'op' in ast:
                op = ast.op
            else:
                op = '='

            if 'arg_1' in ast:
                args = [ast.arg_1, ast.arg_2]
            elif 'equal_comp_args' in ast:
                args = ast.equal_comp_args
            elif 'expr_1' in ast:
                args = [ast.expr_1, ast.expr_2]
            elif 'expr' in ast:
                args = ast.expr
                if not isinstance(args, list):
                    args = [args]
            else:
                raise ValueError(f'No arguments found for comparison rule {rule}')

            arg_strings = [self._ast_to_string_key(arg, **kwargs) for arg in args]  # type: ignore
            if all(arg_string == arg_strings[0] for arg_string in arg_strings):
                expr = self.algebra.TRUE if '=' in op else self.algebra.FALSE  # type: ignore

            elif all(self.number_pattern.match(arg_string) for arg_string in arg_strings):
                arg_numbers = [float(arg_string) for arg_string in arg_strings]
                if op == '=':
                    expr = self.algebra.TRUE if all(arg_number == arg_numbers[0] for arg_number in arg_numbers) else self.algebra.FALSE
                else:
                    expr = self.algebra.TRUE if eval(f'{arg_numbers[0]} {op} {arg_numbers[1]}') else self.algebra.FALSE

            else:
                symbol_name = self._key_to_symbol_str(key)
                # In the case of equality, the number matters (in other cases, two comparisons with different numbers make one redundant)
                if rule != 'scoring_equals_comp':
                    symbol_name = DOUBLE_UNDERSCORE_NUMBER_PATTERN.sub(NUMBER_SUB, symbol_name)

                expr = boolean.Symbol(symbol_name)

        # treat arithmetic operations as logicals for the purpose of finding redundancies
        elif rule in ('scoring_multi_expr', 'scoring_binary_expr'):
            if rule == 'scoring_multi_expr':
                args = ast.expr
                if not isinstance(args, list):
                    args = [args]
            else:
                args = [ast.expr_1, ast.expr_2]

            arg_mappings = [self(arg, **kwargs) for arg in args]  # type: ignore

            non_preference_eval_or_number_found = False
            for arg in args:
                while arg.parseinfo.rule in ('scoring_expr_or_number', 'scoring_expr'):  # type: ignore
                    arg = arg.expr  # type: ignore

                inner_rule = arg.parseinfo.rule  # type: ignore
                if inner_rule not in ('scoring_number_value', 'preference_eval', 'scoring_comparison'):
                    non_preference_eval_or_number_found = True
                    break

            # If anything other than a preference eval or number is found, treat this as a logical
            if non_preference_eval_or_number_found:
                if isinstance(arg_mappings, list) and len(arg_mappings) == 1:
                    arg_mappings = arg_mappings[0]

                if isinstance(arg_mappings, boolean.Expression):
                    expr = arg_mappings

                else:
                    expr_type = boolean.AND  # if self.next_arithmetic_operator_as_and else boolean.OR
                    expr = expr_type(*arg_mappings)
                    # self.next_arithmetic_operator_as_and = not self.next_arithmetic_operator_as_and

            # If only counts and numbers, treat this as a discrete symbol, after filtering out trues/falses
            else:
                filtered_args = []

                for arg in arg_mappings:
                    if arg == self.true:
                        continue

                    if arg == self.false:
                        expr = self.false
                        break

                    if isinstance(arg, boolean.Symbol):
                        filtered_args.append(arg)

                    else:
                        raise ValueError(f'Unexpected argument in scoring expression: {arg} ({type(arg)})')

                if expr is None:
                    if len(filtered_args) > 0:
                        expr = boolean.Symbol('__'.join(symbol.obj for symbol in filtered_args))  # type: ignore
                    else:
                        expr = self.true

        elif rule == 'scoring_neg_expr':
            arg_mapping = self(ast.expr, **kwargs)  # type: ignore
            expr = boolean.NOT(arg_mapping)

        elif rule.startswith('count'):
            expr = boolean.Symbol(key[1:-1].replace(' ', '__'))

        elif rule == 'scoring_number_value':
            expr = boolean.Symbol('NUMBER')

        else:
            keys = set(ast.keys())
            if 'parseinfo' in keys:
                keys.remove('parseinfo')

            if len(keys) == 1:
                expr = self(ast[keys.pop()], **kwargs)  # type: ignore

        if expr is None:
            logger.warn(f'No expression found for rule {rule}, using "{key}"')
            expr = boolean.Symbol(key)

        expr = typing.cast(boolean.Expression, expr)
        self.str_to_expression_mapping[key] = expr
        return expr

    def _ast_to_string_key(self, ast, **kwargs):
        return self.whitespace_pattern.sub(' ', ast_printer.ast_section_to_string(ast, f'(:{kwargs[SECTION_CONTEXT_KEY]}' if not kwargs[SECTION_CONTEXT_KEY].startswith('(:') else kwargs[SECTION_CONTEXT_KEY]))


PREFERENCE_REPLACEMENT_PATTERN = 'preference{index}'
VARIABLE_REPLACEMENT_PATTERN = '?v{index}'


class ASTSamplePostprocessor(ASTParser):
    def __init__(self, preference_pattern: str = PREFERENCE_REPLACEMENT_PATTERN, variable_pattern: str = VARIABLE_REPLACEMENT_PATTERN):
        self.preference_pattern = preference_pattern
        self.variable_pattern = variable_pattern
        self._new_parse()

    def _new_parse(self):
        self.preference_index = 0
        self.variable_index = 0
        self.preference_mapping = {}
        self.variable_mapping = {}
        self.parseinfo_index = 0

    def __call__(self, ast, should_deepcopy_initial: bool = True, **kwargs):
        initial_call = 'inner_call' not in kwargs or not kwargs['inner_call']
        if initial_call:
            kwargs['inner_call'] = True
            self._new_parse()
            if should_deepcopy_initial:
                ast = ast_utils.deepcopy_ast(ast)

        super().__call__(ast, **kwargs)

        if initial_call:
            return ast

    def _handle_ast(self, ast, **kwargs):
        rule = ast.parseinfo.rule  # type: ignore

        new_parseinfo = tatsu.infos.ParseInfo(None, rule, self.parseinfo_index, self.parseinfo_index, self.parseinfo_index, self.parseinfo_index)
        ast_utils.replace_child(ast, 'parseinfo', new_parseinfo)
        self.parseinfo_index += 1

        if rule == 'preference':
            pref_name = ast.pref_name
            if pref_name not in self.preference_mapping:
                self.preference_mapping[pref_name] = self.preference_pattern.format(index=self.preference_index)
                self.preference_index += 1

            ast_utils.replace_child(ast, 'pref_name', self.preference_mapping[pref_name])

        elif rule == 'pref_name_and_types':
            pref_name = ast.pref_name
            if pref_name not in self.preference_mapping:
                pref_name = 'preferenceUndefined'
            else:
                pref_name = self.preference_mapping[pref_name]

            ast_utils.replace_child(ast, 'pref_name', pref_name)

        elif rule == 'variable_type_def':
            var_names = ast.var_names
            if isinstance(var_names, str):
                if var_names not in self.variable_mapping:
                    self.variable_mapping[var_names] = self.variable_pattern.format(index=self.variable_index)
                    self.variable_index += 1

                ast_utils.replace_child(ast, 'var_names', self.variable_mapping[var_names])

            else:
                new_var_names = []
                for var_name in var_names:  # type: ignore
                    if var_name not in self.variable_mapping:
                        self.variable_mapping[var_name] = self.variable_pattern.format(index=self.variable_index)
                        self.variable_index += 1

                    new_var_names.append(self.variable_mapping[var_name])

                ast_utils.replace_child(ast, 'var_names', new_var_names)

        if rule.startswith('predicate_or_function_') and rule.endswith('term'):
            term = ast.term
            if isinstance(term, str) and term.startswith('?'):
                if term not in self.variable_mapping:
                    self.variable_mapping[term] = self.variable_pattern.format(index=self.variable_index)
                    self.variable_index += 1

                ast_utils.replace_child(ast, 'term', self.variable_mapping[term])

        super()._handle_ast(ast, **kwargs)


def _extract_variable_type_as_list(var_type: tatsu.ast.AST) -> typing.List[str]:
    var_type_rule = var_type.parseinfo.rule  # type: ignore

    if var_type_rule.endswith('type_definition'):
        var_type = var_type.type  # type: ignore
        var_type_rule = var_type.parseinfo.rule  # type: ignore

    if var_type_rule.endswith('type'):
        return [var_type.terminal]  # type: ignore

    elif var_type_rule.startswith('either'):
        type_names = var_type.type_names
        if not isinstance(type_names, list):
            type_names = [type_names]

        return [t.terminal for t in type_names]  # type: ignore

    else:
        raise ValueError(f'Unexpected variable type rule: {var_type_rule}')


def extract_variables(predicate: typing.Union[typing.Sequence[tatsu.ast.AST], tatsu.ast.AST, None], error_on_repeat: bool = False) -> typing.List[str]:
    '''
    Recursively extract every variable referenced in the predicate (including inside functions
    used within the predicate)
    '''
    if predicate is None:
        return []

    if isinstance(predicate, list) or isinstance(predicate, tuple):
        pred_vars = []
        for sub_predicate in predicate:
            pred_vars.extend(extract_variables(sub_predicate))

        unique_vars = []
        for var in pred_vars:
            if var not in unique_vars:
                unique_vars.append(var)

        return unique_vars

    elif isinstance(predicate, tatsu.ast.AST):
        pred_vars = []
        exists_forall_vars = []

        for key in predicate:
            if key == "term":
                term = predicate["term"]

                # Different structure for predicate args vs. function args
                if isinstance(term, tatsu.ast.AST):
                    if "terminal" in term:
                        pred_vars.append(term["terminal"])

                    elif "arg" in term:  # comparison argument
                        pred_vars.append(term["arg"])  # type: ignore

                    else:
                        raise ValueError(f'Unexpected predicate term structure in `extract_variables`: {term}')
                else:
                    pred_vars.append(term)

            elif key == "var_names":
                pred_vars.append(predicate["var_names"]) # type: ignore

            # We don't want to capture any variables within an (exists) or (forall) that's inside
            # the preference, since those are not globally required -- see evaluate_predicate()
            elif key == "exists_vars":
                exists_forall_vars += extract_variables(predicate[key])

            elif key == "forall_vars":
                exists_forall_vars += extract_variables(predicate[key])

            elif key != "parseinfo":
                pred_vars.extend(extract_variables(predicate[key]))

        unique_vars = []
        for var in pred_vars:
            if var not in unique_vars and var not in exists_forall_vars:
                unique_vars.append(var)

        if error_on_repeat and predicate.parseinfo.rule == 'predicate' and len(unique_vars) != len(pred_vars):
            raise ValueError(f'Found duplicate variables in predicate: {pred_vars}')

        return unique_vars

    else:
        return []
