import functools
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import var
import tatsu
import tatsu.ast
import typing

import ast_parser


DEFAULT_INCREMENT = '  '
BUFFER = None
LINE_BUFFER = None

PREDICATES = [
    'predicate_above',
    'predicate_adjacent',
    'predicate_adjacent_side_3',
    'predicate_adjacent_side_4',
    'predicate_agent_crouches',
    'predicate_agent_holds',
    'predicate_between',
    'predicate_broken',
    'predicate_equal_x_position',
    'predicate_equal_z_position',
    'predicate_faces',
    'predicate_game_over',
    'predicate_game_start',
    'predicate_in',
    'predicate_in_motion',
    'predicate_is_setup_object',
    'predicate_object_orientation',
    'predicate_on',
    'predicate_open',
    'predicate_opposite',
    'predicate_rug_color_under',
    'predicate_same_color',
    'predicate_same_object',
    'predicate_same_type',
    'predicate_toggled_on',
    'predicate_touch'
]

FUNCTIONS = [
    'function_building_size',
    'function_distance',
    'function_distance_side_3',
    'function_distance_side_4',
    'function_x_position',
]


class ASTPrinter(ast_parser.ASTParser):
    def __init__(self, ast_key, rule_name_substitutions):
        self.ast_key = ast_key
        self.rule_name_substitutions = rule_name_substitutions
        self.exact_matches = {}
        self.keyword_matches = []

    def register_exact_matches(self, *handlers):
        for handler in handlers:
            self.register_exact_match(handler)

    def register_exact_match(self, handler, rule_name=None):
        if rule_name is None:
            rule_name = handler.__name__.replace('_handle_', '')

        if isinstance(rule_name, str):
            self.exact_matches[rule_name] = handler
        else:
            for name in rule_name:
                self.exact_matches[name] = handler

    def register_keyword_match(self, keywords, handler):
        if isinstance(keywords, str):
            keywords = (keywords,)
        self.keyword_matches.append((keywords, handler))

    def _handle_tuple(self, ast, **kwargs):
        _indent_print(ast[0], kwargs['depth'], kwargs['increment'], kwargs['context'])
        self(ast[1], depth=kwargs['depth'] + 1, increment=kwargs['increment'], context=kwargs['context'])
        if len(ast) > 2:
            if len(ast) > 3:
                raise ValueError(f'Unexpectedly long tuple: {ast}')

            _indent_print(ast[2], kwargs['depth'], kwargs['increment'], kwargs['context'])

    def _handle_str(self, ast, **kwargs):
        return _indent_print(ast, kwargs['depth'], kwargs['increment'], kwargs['context'])

    def _handle_int(self, ast, **kwargs):
        return self._handle_str(str(ast), **kwargs)

    def _handle_ast(self, ast, **kwargs):
        rule = ast.parseinfo.rule  # type: ignore
        depth, increment, context = kwargs['depth'], kwargs['increment'], kwargs['context']
        # check for exact matches before rule substitutions

        if rule in self.exact_matches:
            self.exact_matches[rule](self, rule, ast, depth, increment, context)
            return

        # if that didn't work, apply rule substitutions
        for sub in self.rule_name_substitutions:
            rule = rule.replace(sub, '')

        if rule in self.exact_matches:
            self.exact_matches[rule](self, rule, ast, depth, increment, context)
            return

        else:
            for keywords, handler in self.keyword_matches:
                if any([keyword in rule for keyword in keywords]):
                    handler(self, rule, ast, depth, increment, context)
                    return

        raise ValueError(f'No match found in {self.ast_key} for rule: {ast.parseinfo.rule}: {ast}')  # type: ignore


    def __call__(self, ast, depth=0, increment=DEFAULT_INCREMENT, context=None):
        kwargs = dict(depth=depth, increment=increment, context=context)
        return super().__call__(ast, **kwargs)


def reset_buffers(to_list=True):
    global BUFFER, LINE_BUFFER

    if to_list:
        BUFFER = []
        LINE_BUFFER = []

    else:
        BUFFER = None
        LINE_BUFFER = None


def _flush_line_buffer():
    global BUFFER, LINE_BUFFER

    if LINE_BUFFER is None:
        return

    line = ' '.join(LINE_BUFFER)
    if line:
        if BUFFER is None:
            print(line)
        else:
            BUFFER.append(line)


MUTATION_STYLES = {
    'old': {'text-decoration': 'line-through'},
    'new': {'color':  '#FF6A00'},
    'modified': {'color': '#00FF80'},
    'root': {'color': '#AA00FF'}
}
DEFAULT_COLORMAP = plt.get_cmap('tab10')  # type: ignore
MUTATION_STYLES.update({i: {'color': rgb2hex(DEFAULT_COLORMAP(i))} for i in range(10)})  # type: ignore

def preprocess_context(context):
    if not context:
        context = {}

    if not 'html_style' in context:
        context['html_style'] = {}

    if 'color' in context['html_style']: del context['html_style']['color']
    if 'text-decoration' in context['html_style']: del context['html_style']['text-decoration']

    if 'mutation' in context:
        mutation = context['mutation']
        if mutation in MUTATION_STYLES:
            context['html_style'].update(MUTATION_STYLES[mutation])
        elif str(mutation) in MUTATION_STYLES:
            context['html_style'].update(MUTATION_STYLES[str(mutation)])

    return context


def _indent_print(out, depth, increment, context=None):
    global BUFFER, LINE_BUFFER

    context = preprocess_context(context)

    if 'continue_line' in context and context['continue_line'] and LINE_BUFFER is not None:
        LINE_BUFFER.append(out)

    else:
        if LINE_BUFFER:
            if 'html' in context and any([bool(s and not s.isspace()) for s in LINE_BUFFER]):
                LINE_BUFFER.append('</div>')

            _flush_line_buffer()

        if 'html' in context:
            context['html_style']['margin-left'] = f'{20 * depth}px'

            LINE_BUFFER = [f'<div style="{"; ".join({f"{k}: {v}" for k, v in context["html_style"].items()})}">{out}']
        else:
            LINE_BUFFER = [f'{increment * depth}{out}']


def _out_str_to_span(out_str, context, remove: bool = False):
    if 'html' in context and context['html']:
        return f'<span style="{"; ".join({f"{k}: {v}" for k, v in context["html_style"].items() if k != "margin-left"})}">{out_str}</span>'

    if remove:
        return f'<remove>{out_str}</remove>'

    return out_str


def mutation_and_removal_context(func):
    @functools.wraps(func)
    def wrapper_func(caller, rule, ast, depth, increment, context=None, *args, **kwargs):
        if context is None:
            context = {}

        prev_mutation = None
        if 'mutation' in ast:
            if 'mutation' in context:
                prev_mutation = context['mutation']

            context['mutation'] = ast['mutation']

        context = preprocess_context(context)

        should_remove = isinstance(ast, tatsu.ast.AST) and ast.get('remove', False) and not kwargs.get('return_str', False)
        if should_remove:
            _indent_print('<remove>', depth, increment, context)

        ret_val = func(caller, rule, ast, depth, increment, context, *args, **kwargs)

        if should_remove:
            _indent_print('</remove>', depth, increment, context)

        if 'mutation' in ast:
            if prev_mutation is not None:
                context['mutation'] = prev_mutation
            else:
                del context['mutation']

        return ret_val

    return wrapper_func


def _parse_either_types(either_types, context):
    type_names = either_types.type_names
    if not isinstance(type_names, list):
        type_names = [type_names]

    type_names = ' '.join(t.terminal for t in type_names)

    var_type_str = f'(either {type_names})'
    return _out_str_to_span(var_type_str, context, remove=either_types.get('remove', False))


def _handle_either_types(caller, rule, ast, depth, increment, context=None):
    _indent_print(_parse_either_types(ast, context), depth, increment, context)


def validate_variable_list_is_list(func):
    @functools.wraps(func)
    def wrapper_func(caller, rule, var_list, depth, increment, context=None, *args, **kwargs):
        if not isinstance(var_list, list):
            var_list = [var_list]

        return func(caller, rule, var_list, depth, increment, context, *args, **kwargs)

    return wrapper_func

def _parse_type_definition(var_type_def, context):
    var_type = var_type_def.type
    var_type_list = ast_parser._extract_variable_type_as_list(var_type)
    var_type_str = var_type_list[0] if var_type.parseinfo.rule.endswith('type') else f"(either {' '.join(var_type_list)})"
    # var_type_rule = var_type.parseinfo.rule  # type: ignore
    # if var_type_rule.endswith('type'):
    #     var_type_str = var_type.terminal

    # elif var_type_rule.startswith('either'):
    #     var_type_str = _parse_either_types(var_type, context)

    #     inner_prev_mutation = None
    #     if 'html' in context and context['html'] and 'mutation' in var_type:
    #         if 'mutation' in context:
    #             inner_prev_mutation = context['mutation']

    #         context['mutation'] = var_type['mutation']
    #         context = preprocess_context(context)
    #         var_type_str = _out_str_to_span(var_type_str, context)

    #         if inner_prev_mutation is not None:
    #             context['mutation'] = inner_prev_mutation
    #         else:
    #             del context['mutation']
    #         context = preprocess_context(context)

    # else:
    #     raise ValueError(f'Unrecognized quantifier variables: {var_type}')

    return _out_str_to_span(var_type_str, context, remove=var_type_def.get('remove', False))


def _handle_type_definition(caller, rule, ast, depth, increment, context=None):
    _indent_print(_parse_type_definition(ast, context), depth, increment, context)


def _parse_variable_type_def(var_def, context):
    var_names = f'{" ".join(var_def.var_names) if isinstance(var_def.var_names, list) else var_def.var_names}'
    var_type_str = _parse_type_definition(var_def.var_type, context)
    return f'{var_names} - {var_type_str}'


def _handle_variable_type_def(caller, rule, ast, depth, increment, context=None):
    _indent_print(_parse_variable_type_def(ast, context), depth, increment, context)


@validate_variable_list_is_list
@mutation_and_removal_context
def _parse_variable_list(caller, rule, var_list, depth, increment, context=None):
    formatted_vars = []
    context = typing.cast(dict, context)

    for var_def in var_list:
        prev_mutation = None
        if 'mutation' in var_def:
            if 'mutation' in context:
                prev_mutation = context['mutation']

            context['mutation'] = var_def['mutation']

        context = preprocess_context(context)
        var_str = _parse_variable_type_def(var_def, context)
        formatted_vars.append(_out_str_to_span(var_str, context, remove=var_def.get('remove', False)))

        if 'mutation' in var_def:
            if prev_mutation is not None:
                context['mutation'] = prev_mutation
            else:
                del context['mutation']

    return formatted_vars


def _handle_variable_list(caller, rule, ast, depth, increment, context=None):
    out = _parse_variable_list(caller, rule, ast.variables, depth, increment, context)
    _indent_print(_out_str_to_span(f'({" ".join(out)})', context), depth, increment, context)


QUANTIFIER_KEYS = ('args', 'pred', 'then', 'pref')


@mutation_and_removal_context
def _handle_quantifier(caller, rule, ast, depth, increment, context=None):
    context = typing.cast(dict, context)

    prev_continue_line = context['continue_line'] if 'continue_line' in context else False

    _indent_print(_out_str_to_span(f'({rule}', context), depth, increment, context)
    context['continue_line'] = True

    vars_key = None
    args_key = None
    vars_node = None
    var_str = ''
    vars_keys_list = list(filter(lambda k: k.endswith('_vars'), ast.keys()))
    if len(vars_keys_list) > 1:
         raise ValueError(f'Multiple quantifier variables: {ast}')

    if len(vars_keys_list) == 1:
        vars_key = vars_keys_list[0]
        args_key = vars_key.replace('_vars', '_args')
        vars_node = ast[vars_key]
        formatted_vars = _parse_variable_list(caller, rule, vars_node.variables, depth, increment, context)
        var_str = f'({" ".join(formatted_vars)})'

    if vars_node is not None and 'mutation' in vars_node:
        prev_mutation = None
        if 'mutation' in context:
            prev_mutation = context['mutation']
        context['mutation'] = vars_node['mutation']
        context = preprocess_context(context)

        _indent_print(_out_str_to_span(var_str, context, remove=vars_node.get('remove', False)), depth, increment, context)

        if prev_mutation is not None:
            context['mutation'] = prev_mutation
        else:
            del context['mutation']

        context = preprocess_context(context)

    elif vars_node is not None:
        _indent_print(_out_str_to_span(var_str, context, remove=vars_node.get('remove', False)), depth, increment, context)

    context['continue_line'] = prev_continue_line

    found_args = False

    if args_key is not None and args_key in ast:
        found_args = True
        caller(ast[args_key], depth + 1, increment, context)

    if not found_args:
        for key in QUANTIFIER_KEYS:
            key_str = f'{rule}_{key}'
            if key_str in ast:
                found_args = True
                caller(ast[key_str], depth + 1, increment, context)

    if not found_args:
        print(ast.keys())
        print(rule)
        print([f'{rule}_{key}' for key in QUANTIFIER_KEYS])
        raise ValueError(f'Found exists or forall with unknown arugments: {ast}')

    _indent_print(_out_str_to_span(')', context), depth, increment, context)


@mutation_and_removal_context
def _handle_logical(caller, rule, ast, depth, increment, context=None):
    context = typing.cast(dict, context)

    if 'continue_line' in context and context['continue_line'] and 'html' in context and context['html']:
        _indent_print(f'<span style="{"; ".join({f"{k}: {v}" for k, v in context["html_style"].items() if k != "margin-left"})}">', depth, increment, context)

    if f'{rule}_args' in ast:
        _indent_print(f'({rule}', depth, increment, context)
        caller(ast[f'{rule}_args'], depth + 1, increment, context)
    else:
        rule_fragment = rule.split('_')[0]
        key = f'{rule_fragment}_args'
        if key in ast:
            _indent_print(f'({rule_fragment}', depth, increment, context)
            caller(ast[key], depth + 1, increment, context)
        else:
            raise ValueError(f'Found logical with unknown arguments: {ast}')

    _indent_print(f')', depth, increment, context)

    if 'continue_line' in context and context['continue_line'] and 'html' in context and context['html']:
        _indent_print(f'</span>', depth, increment, context)


@mutation_and_removal_context
def _handle_game(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({rule.replace("_", "-")}', depth, increment, context)
    caller(ast[f'{rule.replace("game_", "")}_pred'], depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


@mutation_and_removal_context
def _handle_function_eval(caller, rule, ast, depth, increment, context=None, return_str=False):
    return _handle_predicate(caller, rule, ast, depth, increment, context, return_str=return_str)


# @mutation_and_removal_context  -- handles it internally
def _inline_format_comparison_arg(caller, rule, ast, depth, increment, context=None) -> str:
    arg = ast.arg
    out_str = ''

    if isinstance(arg, tatsu.ast.AST):
        if arg.parseinfo.rule == 'function_eval':  # type: ignore
            # return _inline_format_function_eval(caller, arg.rule, arg, depth, increment, context)
            out_str = _handle_function_eval(caller, arg.rule, arg, depth, increment, context, return_str=True)
        elif arg.parseinfo.rule == 'comparison_arg_number_value':  # type: ignore
            out_str = arg.terminal
        else:
            raise ValueError(f'Unexpected comparison argument: {arg}')
    else:
        out_str = str(arg)

    if ast.get('remove', False):
        out_str = f'<remove>{out_str}</remove>'

    return out_str  # type: ignore


def _handle_comparison_arg(caller, rule, ast, depth, increment, context):
    return _indent_print(_inline_format_comparison_arg(caller, ast.rule, ast, depth, increment, context), depth, increment, context)


@mutation_and_removal_context
def _handle_function_comparison(caller, rule, ast, depth, increment, context=None):
    ast = ast.comp

    comp_op = '='
    if 'comp_op' in ast:
        comp_op = ast.comp_op

    if 'arg_1' in ast:
        args = [_inline_format_comparison_arg(caller, ast.arg_1.rule, ast.arg_1, depth, increment, context),
            _inline_format_comparison_arg(caller, ast.arg_2.rule, ast.arg_2, depth, increment, context)]

    else:
        comp_args = ast.equal_comp_args
        if isinstance(comp_args, tatsu.ast.AST):
            comp_args = [comp_args]

        args = [_inline_format_comparison_arg(caller, arg.rule, arg, depth, increment, context) if isinstance(arg, tatsu.ast.AST) else str(arg)
            for arg in comp_args
        ]

    _indent_print(_out_str_to_span(f'({comp_op} {" ".join(args)})', context, remove=ast.get('remove', False)),
                  depth, increment, context)


@mutation_and_removal_context
def _handle_two_arg_comparison(caller, rule, ast, depth, increment, context=None):
    comp_op = ast.comp_op
    args = [_inline_format_comparison_arg(caller, ast.arg_1.rule, ast.arg_1, depth, increment, context),
            _inline_format_comparison_arg(caller, ast.arg_2.rule, ast.arg_2, depth, increment, context)]

    _indent_print(_out_str_to_span(f'({comp_op} {" ".join(args)})', context, remove=ast.get('remove', False)),
                  depth, increment, context)


@mutation_and_removal_context
def _handle_multiple_args_equal_comparison(caller, rule, ast, depth, increment, context=None):
    comp_op = '='
    comp_args = ast.equal_comp_args
    if isinstance(comp_args, tatsu.ast.AST):
        comp_args = [comp_args]

    args = [_inline_format_comparison_arg(caller, arg.rule, arg, depth, increment, context) if isinstance(arg, tatsu.ast.AST) else str(arg)
        for arg in comp_args
    ]

    _indent_print(_out_str_to_span(f'({comp_op} {" ".join(args)})', context, remove=ast.get('remove', False)),
                  depth, increment, context)


def _extract_predicate_term_value(ast):
    terminal = ast.term
    if isinstance(terminal, tatsu.ast.AST):
        terminal = terminal.terminal

    return terminal


@mutation_and_removal_context
def _handle_predicate_or_function_term(caller, rule, ast, depth, increment, context=None):
    _indent_print(_out_str_to_span(_extract_predicate_term_value(ast), context, remove=ast.get('remove', False)), depth, increment, context)



@mutation_and_removal_context
def _handle_predicate(caller, rule, ast, depth, increment, context, return_str=False,
    child_keys: typing.List[str] = ['pred', 'func'], child_rule_prefixes: typing.List[str] = ['predicate_', 'function_']):

    pred = ast
    for key in child_keys:
        if key in pred:
            pred = ast[key]
            break

    name = pred.parseinfo.rule
    for prefix in child_rule_prefixes:
        name = name.replace(prefix, '')

    if name[-1].isdigit():
        name = name[:-2]

    args = []
    arg_index = 1
    arg_key = f'arg_{arg_index}'
    while arg_key in pred and pred[arg_key] is not None:
        terminal = _extract_predicate_term_value(pred[arg_key])

        if isinstance(pred[arg_key], tatsu.ast.AST) and pred[arg_key].get('remove', False):
            terminal = f'<remove>{terminal}</remove>'

        args.append(terminal)

        arg_index += 1
        arg_key = f'arg_{arg_index}'

    out = _out_str_to_span(f'({name} {" ".join(args)})', context, remove=pred.get('remove', False) or (ast.get('remove', False) and return_str))  # remove in ast handled by decorator

    if return_str:
        return out

    _indent_print(out, depth, increment, context)


@mutation_and_removal_context
def _handle_setup(caller, rule, ast, depth, increment, context=None):
    caller(ast.setup, depth, increment, context)


@mutation_and_removal_context
def _handle_statement(caller, rule, ast, depth, increment, context=None):
    caller(ast.statement, depth, increment, context)


@mutation_and_removal_context
def _handle_super_predicate(caller, rule, ast, depth, increment, context=None):
    caller(ast.pred, depth, increment, context)


def build_setup_printer():
    printer = ASTPrinter('(:setup', ('setup_', 'super_', 'predicate_'))
    printer.register_exact_matches(
        _handle_setup, _handle_statement, _handle_super_predicate,
        _handle_function_comparison, _handle_function_eval, _handle_predicate,
        _handle_two_arg_comparison, _handle_comparison_arg,
        _handle_multiple_args_equal_comparison, _handle_variable_list,
    )
    printer.register_exact_match(_handle_number_value, 'comparison_arg_number_value')
    printer.register_exact_match(_handle_predicate, PREDICATES + FUNCTIONS)
    printer.register_keyword_match('variable_type_def', _handle_variable_type_def)
    printer.register_keyword_match('type_definition', _handle_type_definition)
    printer.register_keyword_match('either_types', _handle_either_types)
    printer.register_keyword_match(('predicate_or_function',), _handle_predicate_or_function_term)
    printer.register_keyword_match(('exists', 'forall'), _handle_quantifier)
    printer.register_keyword_match(('game',), _handle_game)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer


@mutation_and_removal_context
def _handle_preference(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(preference {ast.pref_name}', depth, increment, context)
    caller(ast.pref_body, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)


@mutation_and_removal_context
def _handle_then(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(then', depth, increment, context)
    caller(ast.then_funcs, depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


@mutation_and_removal_context
def _handle_at_end(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(at-end', depth, increment, context)
    caller(ast.at_end_pred, depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


@mutation_and_removal_context
def _handle_always(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(always', depth, increment, context)
    caller(ast.always_pred, depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


@mutation_and_removal_context
def _handle_any(caller, rule, ast, depth, increment, context=None):
    _indent_print('(any)', depth, increment, context)


@mutation_and_removal_context
def _handle_once(caller, rule, ast, depth, increment, context=None):
    context = typing.cast(dict, context)
    _indent_print('(once', depth, increment, context)
    context['continue_line'] = True
    caller(ast.once_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_and_removal_context
def _handle_once_measure(caller, rule, ast, depth, increment, context=None):
    context = typing.cast(dict, context)
    _indent_print('(once-measure', depth, increment, context)
    context['continue_line'] = True
    caller(ast.once_measure_pred, depth + 1, increment, context)
    # _indent_print(_inline_format_function_eval(caller, ast.measurement.rule, ast.measurement, depth, increment, context), depth + 1, increment, context)
    _indent_print(_handle_function_eval(caller, ast.measurement.rule, ast.measurement, depth, increment, context, return_str=True), depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_and_removal_context
def _handle_hold(caller, rule, ast, depth, increment, context=None):
    context = typing.cast(dict, context)
    _indent_print('(hold', depth, increment, context)
    context['continue_line'] = True
    caller(ast.hold_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_and_removal_context
def _handle_while_hold(caller, rule, ast, depth, increment, context=None):
    context = typing.cast(dict, context)
    _indent_print('(hold-while', depth, increment, context)
    context['continue_line'] = True
    caller([ast.hold_pred, ast.while_preds], depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_and_removal_context
def _handle_hold_for(caller, rule, ast, depth, increment, context=None):
    context = typing.cast(dict, context)
    _indent_print(f'(hold-for {ast.num_to_hold}', depth, increment, context)
    context['continue_line'] = True
    caller(ast.hold_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_and_removal_context
def _handle_hold_to_end(caller, rule, ast, depth, increment, context=None):
    context = typing.cast(dict, context)
    _indent_print('(hold-to-end', depth, increment, context)
    context['continue_line'] = True
    caller(ast.hold_pred, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_and_removal_context
def _handle_forall_seq(caller, rule, ast, depth, increment, context=None):
    context = typing.cast(dict, context)
    variables = ast.forall_seq_vars.variables
    formatted_vars = _parse_variable_list(caller, rule, variables, depth, increment, context)
    var_str = f'({" ".join(formatted_vars)})'
    if 'html' in context and context['html'] and 'mutation' in variables:
        prev_mutation = None
        if 'mutation' in context:
            prev_mutation = context['mutation']
        context['mutation'] = ast['mutation']

        _indent_print(f'(forall-sequence {_out_str_to_span(var_str, context, remove=ast.forall_seq_vars.get("remove", False))}', depth, increment, context)

        if prev_mutation is not None:
            context['mutation'] = prev_mutation
        else:
            del context['mutation']

    else:
        _indent_print(f'(forall-sequence {_out_str_to_span(var_str, context, remove=ast.forall_seq_vars.get("remove", False))}', depth, increment, context)

    caller(ast.forall_seq_then, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)


@mutation_and_removal_context
def _handle_preferences(caller, rule, ast, depth, increment, context=None):
    _indent_print('(and', depth, increment, context)
    caller(ast.preferences, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)


@mutation_and_removal_context
def _handle_pref_def(caller, rule, ast, depth, increment, context=None):
    caller(ast.definition, depth, increment, context)


@mutation_and_removal_context
def _handle_pref_body(caller, rule, ast, depth, increment, context=None):
    caller(ast.body, depth, increment, context)


@mutation_and_removal_context
def _handle_seq_func(caller, rule, ast, depth, increment, context=None):
    caller(ast.seq_func, depth, increment, context)


def build_constraints_printer():
    printer = ASTPrinter('(:constraints', ('pref_body_', 'pref_', 'super_', 'predicate_'))
    printer.register_exact_matches(
        _handle_preferences, _handle_pref_def, _handle_pref_body, _handle_seq_func,
        _handle_preference, _handle_super_predicate,
        _handle_function_comparison, _handle_predicate,
        _handle_at_end, _handle_always, _handle_then,
        _handle_any, _handle_once, _handle_once_measure,
        _handle_hold, _handle_while_hold, _handle_hold_for, _handle_hold_to_end,
        _handle_forall_seq, _handle_function_eval, _handle_two_arg_comparison,
        _handle_multiple_args_equal_comparison, _handle_comparison_arg,
        _handle_variable_list,
    )
    printer.register_exact_match(_handle_number_value, 'comparison_arg_number_value')
    printer.register_exact_match(_handle_preferences, 'pref_forall_prefs')
    printer.register_exact_match(_handle_predicate, PREDICATES + FUNCTIONS)
    printer.register_keyword_match('variable_type_def', _handle_variable_type_def)
    printer.register_keyword_match('type_definition', _handle_type_definition)
    printer.register_keyword_match('either_types', _handle_either_types)
    printer.register_keyword_match('predicate_or_function', _handle_predicate_or_function_term)
    printer.register_keyword_match(('exists', 'forall'), _handle_quantifier)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)

    return printer


@mutation_and_removal_context
def _handle_binary_comp(caller, rule, ast, depth, increment, context=None):
    if 'comp' in ast:
        ast = ast.comp

    context = typing.cast(dict, context)
    _indent_print(f'({ast.op}', depth, increment, context)
    context['continue_line'] = True
    caller([ast.expr_1, ast.expr_2], depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_and_removal_context
def _handle_multi_expr(caller, rule, ast, depth, increment, context=None):
    context = typing.cast(dict, context)
    _indent_print(f'({ast.op}', depth, increment, context)
    context['continue_line'] = True
    caller(ast.expr, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_and_removal_context
def _handle_binary_expr(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'({ast.op}', depth, increment, context)
    caller([ast.expr_1, ast.expr_2], depth + 1, increment, context)
    _indent_print(')', depth, increment, context)


@mutation_and_removal_context
def _handle_neg_expr(caller, rule, ast, depth, increment, context=None):
    _indent_print('(-', depth, increment, context)
    if context is None:
        context = {}
    context['continue_line'] = True
    caller(ast.expr, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_and_removal_context
def _handle_equals_comp(caller, rule, ast, depth, increment, context=None):
    _indent_print('(=', depth, increment, context)
    if context is None:
        context = {}
    context['continue_line'] = True
    caller(ast.expr, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


@mutation_and_removal_context
def _handle_with(caller, rule, ast, depth, increment, context=None):
    _indent_print('(with ', depth, increment, context)
    if context is None:
        context = {}
    context['continue_line'] = True
    var_node = ast['with_vars']
    formatted_vars = _parse_variable_list(caller, rule, var_node.variables, depth, increment, context)
    var_str = f'({" ".join(formatted_vars)})'

    if 'mutation' in var_node:
        prev_mutation = None
        if 'mutation' in context:
            prev_mutation = context['mutation']
        context['mutation'] = var_node['mutation']
        context = preprocess_context(context)

        _indent_print(_out_str_to_span(var_str, context), depth, increment, context)

        if prev_mutation is not None:
            context['mutation'] = prev_mutation
        else:
            del context['mutation']

        context = preprocess_context(context)

    else:
        _indent_print(_out_str_to_span(var_str, context), depth, increment, context)

    caller(ast.with_pref, depth + 1, increment, context)
    _indent_print(')', depth, increment, context)
    context['continue_line'] = False


def _parse_pref_object_type(pref_object_type, context):
    type_name = pref_object_type.type_name
    if isinstance(type_name, tatsu.ast.AST):
        type_name = type_name.terminal
    return _out_str_to_span(type_name, context, remove=pref_object_type.get('remove', False))


@mutation_and_removal_context
def _handle_pref_object_type(caller, rule, ast, depth, increment, context=None):
    _indent_print(_parse_pref_object_type(ast, context), depth, increment, context)


def _parse_pref_name_and_types(pref_name_and_types, context):
    type_str = ''
    if pref_name_and_types.object_types:
        types = pref_name_and_types.object_types
        if isinstance(types, tatsu.ast.AST):
            types = [types]

        type_str = ":" + ":".join([_parse_pref_object_type(t, context) for t in types])

    return _out_str_to_span(f'{pref_name_and_types.pref_name}{type_str}', context, remove=pref_name_and_types.get('remove', False))


@mutation_and_removal_context
def _handle_pref_name_and_types(caller, rule, ast, depth, increment, context=None):
    _indent_print(_parse_pref_name_and_types(ast, context), depth, increment, context)


@mutation_and_removal_context
def _handle_count_method(caller, rule, ast, depth, increment, context=None):
    pref_name_and_types_str = _parse_pref_name_and_types(ast.name_and_types, context)
    _indent_print(_out_str_to_span(f'({ast.parseinfo.rule.replace("_", "-")} {pref_name_and_types_str})', context), depth, increment, context)


@mutation_and_removal_context
def _handle_terminal(caller, rule, ast, depth, increment, context=None):
    caller(ast.terminal, depth, increment, context)


@mutation_and_removal_context
def _handle_terminal_expr(caller, rule, ast, depth, increment, context=None):
    caller(ast.expr, depth, increment, context)


@mutation_and_removal_context
def _handle_scoring_expr(caller, rule, ast, depth, increment, context=None):
    caller(ast.expr, depth, increment, context)


@mutation_and_removal_context
def _handle_scoring_expr_or_number(caller, rule, ast, depth, increment, context=None):
    caller(ast.expr, depth, increment, context)


@mutation_and_removal_context
def _handle_scoring_external_maximize(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(external-forall-maximize', depth, increment, context)
    caller(ast.scoring_expr, depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


@mutation_and_removal_context
def _handle_scoring_external_minimize(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'(external-forall-minimize', depth, increment, context)
    caller(ast.scoring_expr, depth + 1, increment, context)
    _indent_print(f')', depth, increment, context)


@mutation_and_removal_context
def _handle_terminal_comp(caller, rule, ast, depth, increment, context=None):
    if 'comp' in ast:
        ast = ast.comp
    _handle_binary_comp(caller, rule, ast, depth, increment, context=None)


@mutation_and_removal_context
def _handle_preference_eval(caller, rule, ast, depth, increment, context=None):
    caller(ast.count_method, depth, increment, context)


def _handle_number_value(caller, rule, ast, depth, increment, context=None):
    _indent_print(ast.terminal, depth, increment, context)


def build_terminal_printer():
    printer = ASTPrinter('(:terminal', ('terminal_', 'scoring_'))
    printer.register_exact_matches(
        _handle_terminal, _handle_terminal_expr, _handle_pref_name_and_types,
        _handle_scoring_expr, _handle_scoring_expr_or_number,
        _handle_preference_eval, _handle_scoring_comparison,
        _handle_scoring_external_maximize, _handle_scoring_external_minimize,
        _handle_multi_expr, _handle_binary_expr,
        _handle_neg_expr, _handle_equals_comp, _handle_terminal_comp,
        _handle_function_eval, _handle_with, _handle_pref_object_type,
    )
    printer.register_exact_match(_handle_binary_comp, 'comp')
    printer.register_exact_match(_handle_number_value, ('time_number_value', 'score_number_value', 'pref_count_number_value', 'scoring_number_value'))
    printer.register_keyword_match(('count',), _handle_count_method)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer


@mutation_and_removal_context
def _handle_maximize(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'maximize', depth, increment, context)
    caller(ast.expr, depth + 1, increment, context)


@mutation_and_removal_context
def _handle_minimize(caller, rule, ast, depth, increment, context=None):
    _indent_print(f'minimize', depth, increment, context)
    caller(ast.expr, depth + 1, increment, context)


@mutation_and_removal_context
def _handle_scoring(caller, rule, ast, depth, increment, context=None):
    caller(ast.scoring, depth, increment, context)


@mutation_and_removal_context
def _handle_scoring_comparison(caller, rule, ast, depth, increment, context=None):
    caller(ast.comp, depth, increment, context)


def build_scoring_printer():
    printer = ASTPrinter('(:scoring', ('scoring_',))
    printer.register_exact_matches(
        _handle_scoring, _handle_maximize, _handle_minimize, _handle_pref_name_and_types,
        _handle_scoring_expr, _handle_scoring_expr_or_number,
        _handle_preference_eval, _handle_scoring_comparison,
        _handle_scoring_external_maximize, _handle_scoring_external_minimize,
        _handle_multi_expr, _handle_binary_expr, _handle_neg_expr, _handle_scoring_expr_or_number,
        _handle_number_value, _handle_equals_comp, _handle_function_eval,
        _handle_with, _handle_pref_object_type,
    )
    printer.register_exact_match(_handle_binary_comp, 'comp')
    printer.register_keyword_match(('count',), _handle_count_method)
    printer.register_keyword_match(('and', 'or', 'not'), _handle_logical)
    return printer



PARSE_DICT = {
    ast_parser.SETUP: build_setup_printer(),
    ast_parser.PREFERENCES: build_constraints_printer(),
    ast_parser.TERMINAL: build_terminal_printer(),
    ast_parser.SCORING: build_scoring_printer(),
}


def pretty_print_ast(ast, increment=DEFAULT_INCREMENT, context=None):
    _indent_print(f'{ast[0]} (game {ast[1]["game_name"]}) (:domain {ast[2]["domain_name"]})', 0, increment, context)
    ast = ast[3:]

    while ast:
        key = ast[0][0]

        if key == ')':
            _indent_print(f')', 0, increment, context)
            ast = None

        elif key in PARSE_DICT:
            PARSE_DICT[key](ast[0], 0, increment, context)
            ast = ast[1:]

        else:
            print(f'Encountered unknown key: {key}\n')

    _indent_print('', 0, increment, context)


def _postprocess_ast_to_string(ast_str: str):
    return ast_str.replace(' )', ')')


def ast_to_lines(ast: tatsu.ast.AST, increment=DEFAULT_INCREMENT, context=None) -> typing.List[str]:
    reset_buffers(to_list=True)
    pretty_print_ast(ast, increment=increment, context=context)
    _flush_line_buffer()
    return BUFFER  # type: ignore


def ast_to_string(ast: tatsu.ast.AST, line_delimiter: str = '', increment=DEFAULT_INCREMENT, context=None, postprocess: bool = True) -> str:
    out_str = line_delimiter.join(ast_to_lines(ast, increment, context)).strip()  # type: ignore
    if postprocess:
        return _postprocess_ast_to_string(out_str)

    return out_str


def ast_section_to_string(ast: tatsu.ast.AST, section_key: str, line_delimiter: str = '', context=None, postprocess: bool = True) -> str:
    if context is None:
        context = {}
    reset_buffers(to_list=True)
    PARSE_DICT[section_key](ast, context=context)
    _flush_line_buffer()
    out_str = line_delimiter.join(BUFFER).strip()  # type: ignore
    if postprocess:
        return _postprocess_ast_to_string(out_str)

    return out_str
