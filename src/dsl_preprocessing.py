import argparse
import tatsu

import ast_printer
from ast_parser import ASTParser, ASTParentMapper
from ast_utils import copy_ast, load_asts, replace_child


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_TEST_FILES = (
    # '../dsl/problems-few-objects.pddl',
    # '../dsl/problems-medium-objects.pddl',
    # '../dsl/problems-many-objects.pddl',
    './dsl/interactive-beta.pddl',
)
parser.add_argument('-t', '--test-files', action='append', default=[])
DEFAULT_OUTPUT_PATH ='./preprocessing_examples.csv'
parser.add_argument('-o', '--output-path', default=DEFAULT_OUTPUT_PATH)
parser.add_argument('-p', '--print-dsls', action='store_true')


PREPROCESS_SUBSTITUTIONS = {
    'floor': 'floor_obj',
    'desk': 'desk_obj',
    'chair': 'chair_obj',
}

class ASTPreprocessor(ASTParser):
    def __init__(self, preprocess_substitutions=PREPROCESS_SUBSTITUTIONS):
        self.preprocess_substitutions = preprocess_substitutions

    def __call__(self, ast, **kwargs):
        self._default_kwarg(kwargs, 'parent_mapping', lambda: ASTParentMapper()(ast), should_call=True)
        self._default_kwarg(kwargs, 'local_substitutions', {})
        super().__call__(ast, **kwargs)
        return ast

    def _handle_ast(self, ast, **kwargs):
        parent_mapping, local_substitutions = kwargs['parent_mapping'], kwargs['local_substitutions']

        args_key = None

        if 'pred_args' in ast:
            args_key = 'pred_args'
        elif 'func_args' in ast:
            args_key = 'func_args'

        if args_key is not None:
            for i, arg in enumerate(ast[args_key]):
                if isinstance(arg, str):
                    if arg in local_substitutions:
                        ast[args_key][i] = local_substitutions[arg]
                else:
                    self(arg, **kwargs)

        else:
            vars_keys = [key for key in ast.keys() if key.endswith('_vars')]
            if len(vars_keys) > 1:
                raise ValueError(f'Found multiple variables keys: {vars_keys}', ast)

            elif len(vars_keys) > 0:
                vars_key = vars_keys[0]
                args_keys = [key for key in ast.keys() if key.startswith(vars_key.replace('_vars', ''))]
                args_keys.remove(vars_key)

                if len(args_keys) > 1:
                    raise ValueError(f'Found too many argument keys under: {args_keys}', ast, ast.keys())

                args_key = args_keys[0]

                local_subs, remove_quantifier = self._extract_substitutions_from_vars(ast, vars_key)
                local_substitutions.update(local_subs)

                self(ast[args_key], **kwargs)

                if remove_quantifier:
                    _, parent, selector = parent_mapping[ast.parseinfo]
                    replace_child(parent, selector, ast[args_key])

                for key in local_subs:
                    if key in local_substitutions:
                        del local_substitutions[key]

            else:
                for key in ast:
                    if key != 'parseinfo':
                        self(ast[key], **kwargs)

    def _extract_substitutions_from_vars(self, ast, vars_key):
        substitutions = {}
        var_defs_to_remove = []
        for var_def in ast[vars_key].variables:
            # TODO: what do I do if something being substitued is part of an (either ...)?
            # answer to the above question is nothing, since it wouldn't make sense to eliminate the variable
            # the first if catches that -- if it's an either, var_def.var_type is not a string.
            if isinstance(var_def.var_type, str) and var_def.var_type in self.preprocess_substitutions:
                var_defs_to_remove.append(var_def)
                for name in var_def.var_names:
                    substitutions[name] = self.preprocess_substitutions[var_def.var_type]

        [ast[vars_key].variables.remove(var_def) for var_def in var_defs_to_remove]

        return substitutions, len(ast[vars_key].variables) == 0


def main(args):
    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    asts = load_asts(args, grammar_parser, should_print=args.print_dsls)
    preprocessor = ASTPreprocessor()

    for ast in asts:
        ast_copy = copy_ast(grammar_parser, ast)
        # preprocess_ast_recursive(processed_ast, preprocess_substitutions)
        # processed_ast = preprocess_ast(grammar_parser, ast)
        processed_ast = preprocessor(ast_copy)
        ast_printer.reset_buffers(False)
        ast_printer.pretty_print_ast(processed_ast, context=dict())
        print('\r\n' + '=' * 100 + '\r\n')


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    main(args)
