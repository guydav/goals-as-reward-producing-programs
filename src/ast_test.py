from parse_dsl import *
grammar = tatsu.compile(open('../dsl/dsl.ebnf').read())
DEFAULT_TEST_FILES = (
    '../dsl/problems-few-objects.pddl',
    '../dsl/problems-medium-objects.pddl',
    '../dsl/problems-many-objects.pddl',
    '../dsl/interactive-beta.pddl',
)

asts = {}
game_id_to_ast = {}

for test_file in DEFAULT_TEST_FILES:
    first_dash_idx = test_file.find('-') + 1
    second_dash_idx = test_file.find('-', first_dash_idx)
    name = test_file[first_dash_idx:second_dash_idx]

    test_cases = load_games_from_file(test_file)

    asts[name] = [grammar.parse(case) for case in test_cases]
    game_id_to_ast[name] = {int(ast[1].game_name[ast[1].game_name.rfind('-') + 1:]): ast for ast in asts[name]}


# def get_ast(i):
#     return grammar.parse(test_cases[i])
