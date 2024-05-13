import argparse
from collections import namedtuple, defaultdict, Counter
import copy
from dataclasses import dataclass
from functools import reduce, wraps
import gzip
import itertools
import logging
import multiprocessing
from multiprocessing import pool as mpp
import pickle
import re
import typing

import numpy as np
import pandas as pd
import tatsu
import tatsu.ast
import tatsu.exceptions
import tatsu.infos
import tatsu.grammars
from tatsu import grammars
import tqdm

import ast_printer
from ast_utils import cached_load_and_parse_games_from_file, replace_child, fixed_hash, load_games_from_file, simplified_context_deepcopy, deepcopy_ast
from ast_parser import ASTParser, ASTParentMapper, ASTDepthParser, SECTION_KEYS, PREFERENCES, ContextDict, ASTParentMapping, LOCAL_CONTEXT_PROPAGATING_RULES
import ast_context_fixer
import room_and_object_types
from room_and_object_types import COLOR, ORIENTATION, SIDE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
DEFAULT_COUNTER_OUTPUT_PATH ='./data/ast_counter.pickle'
parser.add_argument('--counter-output-path', default=DEFAULT_COUNTER_OUTPUT_PATH)
DEFAULT_SAMPLES_OUTPUT_PATH = './dsl/ast-mle-samples.pddl'
parser.add_argument('--samples-output-path', default=DEFAULT_SAMPLES_OUTPUT_PATH)
parser.add_argument('-s', '--save-samples', action='store_true')
parser.add_argument('-c', '--parse-counter', action='store_true')
parser.add_argument('--relative-path', type=str, default=None)
parser.add_argument('-n', '--num-samples', type=int, default=10)
parser.add_argument('-p', '--print-samples', action='store_true')
parser.add_argument('-v', '--validate-samples', action='store_true')
parser.add_argument('--sample-tqdm', action='store_true')
parser.add_argument('--inner-sample-tqdm', action='store_true')
DEFAULT_RANDOM_SEED = 33
parser.add_argument('--random-seed', type=int, default=DEFAULT_RANDOM_SEED)
DEFAULT_RECURSION_LIMIT = 3000
parser.add_argument('--recursion-limit', type=int, default=DEFAULT_RECURSION_LIMIT)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--file-open-mode', default='wt')
parser.add_argument('--regrowth-start-index', type=int, default=0)
parser.add_argument('--regrowth-end-index', type=int, default=-1)
parser.add_argument('--section-sample-weights-key', type=str, default=None)
parser.add_argument('--depth-weight-function-key', type=str, default=None)
parser.add_argument('--prior-count', action='append', type=int, default=[])
parser.add_argument('--fix-contexts', action='store_true')
parser.add_argument('--min-n-regrowths', type=int, default=1)
parser.add_argument('--max-n-regrowths', type=int, default=1)
# parser.add_argument('--max-n-attempts', type=int, default=1024)

parser.add_argument('--sample-parallel', action='store_true')
parser.add_argument('--parallel-n-workers', type=int, default=8)
parser.add_argument('--parallel-chunksize', type=int, default=1)
parser.add_argument('--parallel-maxtasksperchild', type=int, default=None)


DEFAULT_MAX_SAMPLE_DEPTH = 16  # 24  # deeper than the deepest game, which has depth 23, and this is for a single node regrowth
parser.add_argument('--max-sample-depth', type=int, default=DEFAULT_MAX_SAMPLE_DEPTH)
DEFAULT_MAX_SAMPLE_NODES = 128  # 256  # longer than most games, but limiting a single node regrowth, not an entire game
parser.add_argument('--max-sample-nodes', type=int, default=DEFAULT_MAX_SAMPLE_NODES)

MLE_SAMPLING = 'mle'
REGROWTH_SAMPLING = 'regrowth'
MCMC_REGRWOTH = 'mcmc-regrowth'
parser.add_argument('--sampling-method', choices=[MLE_SAMPLING, REGROWTH_SAMPLING, MCMC_REGRWOTH], required=True)


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)  # type: ignore
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,  # type: ignore
                                          mpp.starmapstar,  # type: ignore
                                          task_batches),
            result._set_length  # type: ignore
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap  # type: ignore


class SamplingException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class RuleKeyValueCounter:
    def __init__(self, rule: str, key:str):
        self.rule = rule
        self.key = key
        self.rule_counts = defaultdict(int)
        self.value_counts = defaultdict(int)
        self.length_counts = []

    def __call__(self, value: typing.Union[str, typing.Sequence, tatsu.ast.AST]):
        if isinstance(value, str):
            self.value_counts[value] += 1

        elif isinstance(value, list):
            self.length_counts.append(len(value))

        elif isinstance(value, tatsu.ast.AST):
            self.rule_counts[value.parseinfo.rule] += 1  # type: ignore

    def __repr__(self):
        return f'<Counter for {self.rule}.{self.key}: {sum(self.rule_counts.values())} rules counted | {sum(self.value_counts.values())} values counted | {len(self.length_counts)} lengths counted>'


def empty_defaultdict_dict():
    return defaultdict(dict)


class ASTRuleValueCounter(ASTParser):
    """
    This class counts the values appearing for each rule in the AST.
    For the current node, we iterate through all children:
    - If the child is an AST, we count the rule of the AST
    - If the child is a list, we note the length of the list, and count each child
    - If the child is a tuple, TBD, we probably only count some of the elements,
        but need to figure out which ones, which might vary by rule
    """
    def __init__(self):
        self.counters = defaultdict(dict)
        self.counters_by_section = defaultdict(empty_defaultdict_dict)
        self.section_counts = defaultdict(int)

    def __getitem__(self, key):
        return self.counters[key]

    def __call__(self, ast, **kwargs):
        if 'is_root' not in kwargs:
            kwargs['is_root'] = False
            kwargs['section'] = 'preamble'

        return super().__call__(ast, **kwargs)

    def _handle_value(self, ast, **kwargs):
        if 'parent_rule' in kwargs and kwargs['parent_rule'] is not None:
            self.counters[kwargs['parent_rule']][kwargs['rule_key']](ast)
            self.counters_by_section[kwargs['section']][kwargs['parent_rule']][kwargs['rule_key']](ast)

    def _handle_ast(self, ast, **kwargs):
        self._handle_value(ast, **kwargs)
        rule = ast.parseinfo.rule  # type: ignore

        for key in ast:
            if key != 'parseinfo':
                if key not in self.counters[rule]:
                    self.counters[rule][key] = RuleKeyValueCounter(rule, key)

                if key not in self.counters_by_section[kwargs['section']][rule]:
                    self.counters_by_section[kwargs['section']][rule][key] = RuleKeyValueCounter(rule, key)

                new_kwargs = kwargs.copy()
                new_kwargs['parent_rule'] = rule
                new_kwargs['rule_key'] = key
                self(ast[key], **new_kwargs)

    def _handle_str(self, ast, **kwargs):
        self._handle_value(ast, **kwargs)

    def _handle_int(self, ast, **kwargs):
        self._handle_value(ast, **kwargs)

    def _handle_list(self, ast, **kwargs):
        self._handle_value(ast, **kwargs)
        super()._handle_list(ast, **kwargs)

    def _handle_tuple(self, ast, **kwargs):
        if ast[0].startswith('(:'):
            section = ast[0][2:]
            kwargs['section'] = section
            self.section_counts[section] += 1

        return super()._handle_tuple(ast, **kwargs)


def _number_prior(number_list: typing.List):
    return ['0' if np.allclose(x, 0) else (f'{x:.1f}' if isinstance(x, float) else str(x)) for x in set(number_list)]


BINARY_OPERATORS = ['-', '/']
MULTI_OPERATORS = ['+', '*']
COMPARISON_OPERATORS = ['<=', '<', '=', '>=', '>']

COMPARISON_ARG_NUMBER_VALUE_DEFAULTS = _number_prior(list(range(1, 11))) # + list(np.round(np.arange(0, 1, 0.1), 1)))
TIME_NUMBER_VALUE_DEFAULTS = _number_prior(list(range(30, 360, 30)))
SCORE_NUMBER_VALUE_DEFAULTS = _number_prior(list(range(-10, 11)) + list(range(-50, 110, 10)))
PREF_COUNT_NUMBER_VALUE_DEFAULTS = _number_prior(list(range(1, 21)))
SCORING_NUMBER_VALUE_DEFAULTS = _number_prior(list(range(-10, 21)) + list(range(30, 110, 10)) + list(np.round(np.arange(0, 1, 0.1), 1)))

TYPE_NAMES = []
FUNCTION_NAMES = []
PREDICATE_NAMES = []


def _split_posterior(posterior_dict: typing.Dict[typing.Any, typing.Any]):
    return zip(*posterior_dict.items())


def posterior_dict_sample(rng: np.random.Generator, posterior_dict: typing.Dict[str, float], size: typing.Union[int, typing.Tuple[int], None] = None):
    values, probs = _split_posterior(posterior_dict)
    return rng.choice(values, size=size, p=probs)


def _merge_dicts(d1: typing.Dict[typing.Any, typing.Any], d2: typing.Dict[typing.Any, typing.Any]):
    return {**d1, **d2}


def _combine_context_updates(context_updates: typing.List[typing.Optional[ContextDict]]):
    return reduce(_merge_dicts, [d for d in context_updates if d is not None and isinstance(d, dict)], {})


def generate_game_id(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
    game_id = global_context['original_game_id'] if 'original_game_id' in global_context else 'game-id'
    if 'sample_id' in global_context:
        game_id = f'{game_id}-{global_context["sample_id"]}'
    return game_id


DOMAINS = ('few-objects-room-v1', 'medium-objects-room-v1', 'many-objects-room-v1',)


def generate_domain_name(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
    if 'rng' not in global_context:
        rng = np.random.default_rng()
    else:
        rng = global_context['rng']
    return rng.choice(DOMAINS)


class NewPreferenceNameSampler:
    value_posterior: typing.Dict[str, int]

    def __init__(self, value_posterior: typing.Dict[str, int]):
        self.value_posterior = value_posterior

    def __call__(self, global_context: ContextDict, local_context: typing.Optional[ContextDict] = None):
        if 'preference_names' not in global_context:
            global_context['preference_names'] = set()

        filtered_posteior = {k: v for k, v in self.value_posterior.items() if k not in global_context['preference_names']}

        if len(filtered_posteior) == 0:
            raise SamplingException('Attempted to sample a preference name with no available names')

        filtered_posterior_normalization = sum(filtered_posteior.values())
        filtered_posteior = {k: v / filtered_posterior_normalization for k, v in filtered_posteior.items()}

        pref_name = posterior_dict_sample(global_context['rng'], filtered_posteior)

        if isinstance(global_context['preference_names'], dict):
            global_context['preference_names'][pref_name] = 1
        else:
            global_context['preference_names'].add(pref_name)
        return pref_name


def sample_new_preference_name_factory(field_counter: RuleKeyValueCounter, prior_token_count: int=0):
    value_posterior = {k: v + prior_token_count for k, v in field_counter.value_counts.items()}
    return NewPreferenceNameSampler(value_posterior)


sample_new_preference_name_factory.factory = True


def sample_existing_preference_name(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
    if 'preference_names' not in global_context or len(global_context['preference_names']) == 0:
        raise SamplingException('Attempted to sample a preference name with no sampled preferences')

    if 'rng' not in global_context:
        rng = np.random.default_rng()
    else:
        rng = global_context['rng']

    pref_names = list(global_context['preference_names'])
    return rng.choice(pref_names)


class NewVariableSampler:
    total_count: int
    value_posterior: typing.Dict[str, float]

    def __init__(self, total_count: int, value_posterior: typing.Dict[str, float]):
        self.total_count = total_count
        self.value_posterior = value_posterior

    def __call__(self, global_context: ContextDict, local_context: typing.Optional[ContextDict] = None):
        if local_context is None:
            local_context = {}

        if 'variables' not in local_context:
            local_context['variables'] = dict()

        if 'rng' not in global_context:
            rng = np.random.default_rng()
        else:
            rng = global_context['rng']

        valid_vars = set(self.value_posterior.keys()) - set(local_context['variables'].keys())

        if len(valid_vars) == 0:
            raise SamplingException('No valid variables left to sample')

        filtered_posterior_normalization = sum(self.value_posterior[k] for k in valid_vars)
        filtered_posterior = {k: v / filtered_posterior_normalization for k, v in self.value_posterior.items() if k in valid_vars}

        new_var = posterior_dict_sample(rng, filtered_posterior)[1:]
        local_context['variables'][new_var] = None
        return f'?{new_var}'


class SingleLetterVariableSampler:
    key: str
    letter: str

    def __init__(self, letter: str, key_prefix: str):
        self.letter = letter
        self.key = f'{key_prefix}_variables'


class SingleLetterNewVariableSampler(SingleLetterVariableSampler):
    def __call__(self, global_context: ContextDict, local_context: typing.Optional[ContextDict] = None):
        if local_context is None:
            local_context = {}

        if self.key not in local_context:
            local_context[self.key] = dict()

        if self.letter not in local_context[self.key]:
            local_context[self.key][self.letter] = None
            return f'?{self.letter}'

        number = 0
        while f'{self.letter}{number}' in local_context[self.key]:
            number += 1

        local_context[self.key][f'{self.letter}{number}'] = None
        return f'?{self.letter}{number}'


class SingleLetterExistingVariableSampler(SingleLetterVariableSampler):
    def __call__(self, global_context: ContextDict, local_context: typing.Optional[ContextDict] = None):
        if local_context is None:
            local_context = {}

        if self.key not in local_context:
            raise SamplingException(f'Attempted to sample an existing variable with no {self.key} in scope')

        valid_variables = [k for k in local_context[self.key].keys() if k[0] == self.letter]

        if not valid_variables:
            raise SamplingException(f'Attempted to sample an existing variable with no {self.letter} variables (under {self.key}) in scope')

        if 'rng' not in global_context:
            rng = np.random.default_rng()
        else:
            rng = global_context['rng']

        return f'?{rng.choice(valid_variables)}'


def sample_new_variable_factory(field_counter: RuleKeyValueCounter, prior_token_count: int=0):
    total_count = sum(field_counter.value_counts.values()) + (prior_token_count * len(field_counter.value_counts))
    value_posterior = {k: (v + prior_token_count) / total_count for k, v in field_counter.value_counts.items()}
    return NewVariableSampler(total_count, value_posterior)


sample_new_variable_factory.factory = True


def sample_existing_variable(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
    if local_context is None:
        local_context = {}

    if 'variables' not in local_context:
        raise SamplingException('Attempted to sample an existing variable with no variables in scope')

    if 'rng' not in global_context:
        rng = np.random.default_rng()
    else:
        rng = global_context['rng']

    return f'?{rng.choice(list(local_context["variables"].keys()))}'


def sample_empty_list(global_context: ContextDict, local_context: typing.Optional[ContextDict]=None):
    return list()


def create_sample_existing_variable():
    return sample_existing_variable


VARIABLE_DEFAULTS = defaultdict(create_sample_existing_variable)
VARIABLE_DEFAULTS[('variable_type_def', 'var_names')] = sample_new_variable_factory   # type: ignore


COLOR_VARIABLE_LETTER = 'x'
ORIENTATION_VARIABLE_LETTER = 'y'
SIDE_VARIABLE_LETTER = 'z'


def create_color_variable_sampler():
    return SingleLetterExistingVariableSampler(COLOR_VARIABLE_LETTER, COLOR)

def create_orientation_variable_sampler():
    return SingleLetterExistingVariableSampler(ORIENTATION_VARIABLE_LETTER, ORIENTATION)

def create_side_variable_sampler():
    return SingleLetterExistingVariableSampler(SIDE_VARIABLE_LETTER, SIDE)


COLOR_VARIABLE_DEFAULTS = defaultdict(create_color_variable_sampler)
COLOR_VARIABLE_DEFAULTS[('color_variable_type_def', 'var_names')] = SingleLetterNewVariableSampler(COLOR_VARIABLE_LETTER, COLOR)   # type: ignore

ORIENTATION_VARIABLE_DEFAULTS = defaultdict(create_orientation_variable_sampler)
ORIENTATION_VARIABLE_DEFAULTS[('orientation_variable_type_def', 'var_names')] = SingleLetterNewVariableSampler(ORIENTATION_VARIABLE_LETTER, ORIENTATION)   # type: ignore

SIDE_VARIABLE_DEFAULTS = defaultdict(create_side_variable_sampler)
SIDE_VARIABLE_DEFAULTS[('side_variable_type_def', 'var_names')] = SingleLetterNewVariableSampler(SIDE_VARIABLE_LETTER, SIDE)   # type: ignore



COLORS = room_and_object_types.CATEGORIES_TO_TYPES[room_and_object_types.COLORS]
ORIENTATIONS = room_and_object_types.CATEGORIES_TO_TYPES[room_and_object_types.ORIENTATIONS]
SIDES = room_and_object_types.CATEGORIES_TO_TYPES[room_and_object_types.SIDES]

def _comparison_operators():
    return COMPARISON_OPERATORS

def _comparison_operators_no_equal():
    return [c for c in COMPARISON_OPERATORS if c != '=']

def _binary_operators():
    return BINARY_OPERATORS

def _multi_operators():
    return MULTI_OPERATORS

def _function_names():
    return FUNCTION_NAMES


VALID_COLORS = list(COLORS)
if COLOR in VALID_COLORS: VALID_COLORS.remove(COLOR)
VALID_ORIENTATIONS = list(ORIENTATIONS)
if ORIENTATION in VALID_ORIENTATIONS: VALID_ORIENTATIONS.remove(ORIENTATION)
VALID_SIDES = list(SIDES)
if SIDE in VALID_SIDES: VALID_SIDES.remove(SIDE)

# def _colors():
#     l = list(COLORS)
#     l.remove(COLOR)
#     return l

# def _orientations():
#     l = list(ORIENTATIONS)
#     l.remove(ORIENTATION)
#     return l

# def _sides():
#     l = list(SIDES)
#     l.remove(SIDE)
#     return l

def _color_type():
    return [COLOR]

def _orientation_type():
    return [ORIENTATION]

def _side_type():
    return [SIDE]

def _predicate_names():
    return PREDICATE_NAMES

# def _number_defaults():
#     return NUMBER_DEFAULTS

# def _comparison_arg_number_value_defaults():
#     return COMPARISON_ARG_NUMBER_VALUE_DEFAULTS

# def _time_number_value_defaults():
#     return TIME_NUMBER_VALUE_DEFAULTS

# def _score_number_value_defaults():
#     return SCORE_NUMBER_VALUE_DEFAULTS

# def _pref_count_number_value_defaults():
#     return PREF_COUNT_NUMBER_VALUE_DEFAULTS

# def _scoring_number_value_defaults():
#     return SCORING_NUMBER_VALUE_DEFAULTS

def _total_time_defaults():
    return  ['(total-time)']

def _total_score_defaults():
    return  ['(total-score)']

def _directly_named_objects():
    return room_and_object_types.DIRECTLY_REFERRED_OBJECTS


SKIP_CATEGORIES = [room_and_object_types.COLORS, room_and_object_types.ORIENTATIONS, room_and_object_types.SIDES,
                   room_and_object_types.AGENT, room_and_object_types.EMPTY_OBJECT]

ALL_OBJECT_TYPES = set(reduce(lambda x, y: x + y, room_and_object_types.CATEGORIES_TO_TYPES.values()))
ALL_OBJECT_TYPES.update(room_and_object_types.DIRECTLY_REFERRED_OBJECTS)
for skip_category in SKIP_CATEGORIES:
    ALL_OBJECT_TYPES.difference_update(room_and_object_types.CATEGORIES_TO_TYPES[skip_category])

ALL_OBJECT_TYPES = list(ALL_OBJECT_TYPES)


def _all_object_types():
    return ALL_OBJECT_TYPES


DEFAULT_PATTERN_RULE_OPTIONS_BY_RULE = dict(
    binary_comp=defaultdict(_comparison_operators),
    binary_comp_no_equal=defaultdict(_comparison_operators_no_equal),
    binary_op=defaultdict(_binary_operators),
    multi_op=defaultdict(_multi_operators),
    func_name=defaultdict(_function_names),
    name={
        ('object_type', 'terminal'): ALL_OBJECT_TYPES,
        ('object_name', 'terminal'): room_and_object_types.DIRECTLY_REFERRED_OBJECTS,
        ('color_type', 'terminal'): [COLOR],
        ('color', 'terminal'): VALID_COLORS,
        ('orientation_type', 'terminal'): [ORIENTATION],
        ('orientation', 'terminal'): VALID_ORIENTATIONS,
        ('side_type', 'terminal'): [SIDE],
        ('side', 'terminal'): VALID_SIDES,
    },
    # object_type=defaultdict(_all_object_types),
    # object_name=defaultdict(_directly_named_objects),
    # color=defaultdict(_colors),
    # orientation=defaultdict(_orientations),
    # side=defaultdict(_sides),
    # color_type=defaultdict( _color_type),
    # orientation_type=defaultdict(_orientation_type),
    # side_type=defaultdict(_side_type),
    id={
        ('game_def', 'game_name'): generate_game_id,
        ('domain_def', 'domain_name'): generate_domain_name,
    },
    number_pattern={
        ('comparison_arg_number_value', 'terminal'): COMPARISON_ARG_NUMBER_VALUE_DEFAULTS,
        ('time_number_value', 'terminal'): TIME_NUMBER_VALUE_DEFAULTS,
        ('score_number_value', 'terminal'): SCORE_NUMBER_VALUE_DEFAULTS,
        ('pref_count_number_value', 'terminal'): PREF_COUNT_NUMBER_VALUE_DEFAULTS,
        ('scoring_number_value', 'terminal'): SCORING_NUMBER_VALUE_DEFAULTS,
    },
    predicate_name=defaultdict(_predicate_names),
    preference_name={
        ('preference', 'pref_name'): sample_new_preference_name_factory,
        ('pref_name_and_types', 'pref_name'): sample_existing_preference_name,
    },
    variable=VARIABLE_DEFAULTS,
    color_variable=COLOR_VARIABLE_DEFAULTS,
    orientation_variable=ORIENTATION_VARIABLE_DEFAULTS,
    side_variable=SIDE_VARIABLE_DEFAULTS,
    total_time=defaultdict(_total_time_defaults),
    total_score=defaultdict(_total_score_defaults),
)


SPECIAL_RULE_FIELD_VALUE_TYPES = {
    # ('type_definition', 'type'): 'object_type',
    # ('color_type_definition', 'type'): 'color_type',
    # ('orientation_type_definition', 'type'): 'orientation_type',
    # ('side_type_definition', 'type'): 'side_type',
    # ('comparison_arg', 'arg'): 'number',
    ('object_type', 'terminal'): 'name',
    ('object_name', 'terminal'): 'name',
    ('color_type', 'terminal'): 'name',
    ('color', 'terminal'): 'name',
    ('orientation_type', 'terminal'): 'name',
    ('orientation', 'terminal'): 'name',
    ('side_type', 'terminal'): 'name',
    ('side', 'terminal'): 'name',
    # ('predicate_or_function_term', 'term'): ('object_name', 'variable',),
    # ('predicate_or_function_color_term', 'term'): ('color', 'color_variable',),
    # ('predicate_or_function_orientation_term', 'term'): ('orientation', 'orientation_variable',),
    # ('predicate_or_function_side_term', 'term'): ('side', 'side_variable',),
    # ('predicate_or_function_type_term', 'term'): ('object_type', 'variable',),
    ('predicate_or_function_term', 'term'): 'variable',
    ('predicate_or_function_color_term', 'term'): 'color_variable',
    ('predicate_or_function_orientation_term', 'term'): 'orientation_variable',
    ('predicate_or_function_side_term', 'term'): 'side_variable',
    ('predicate_or_function_type_term', 'term'): 'variable',
    ('terminal_expr', 'expr'): ('total_time', 'total_score'),
    # ('scoring_expr_or_number', 'expr'): 'number',
    # ('pref_object_type', 'type_name'): ('object_name', 'object_type'),
}


PATTERN_TYPE_MAPPINGS = {
    'object_name': 'name',
    'object_type': 'name',
    'color': 'name',
    'location': 'name',
    'orientation': 'name',
    'side': 'name',
    'func_name': 'name',
    'preference_name': 'name',
}


SINGLE_LITERAL_RULES = set([
    'color_type',
    'orientation_type',
    'side_type',
    'total_time',
    'total_score'
])


PRIOR_COUNT = 5
LENGTH_PRIOR = {i: PRIOR_COUNT for i in range(5)}

PRODUCTION = 'production'
OPTIONS = 'options'
SAMPLERS = 'samplers'
TYPE_POSTERIOR = 'type_posterior'
RULE_POSTERIOR = 'rule_posterior'
TOKEN_POSTERIOR = 'token_posterior'
LENGTH_POSTERIOR = 'length_posterior'
PRODUCTION_PROBABILITY = 'production_probability'
START = 'start'
EOF = 'EOF'
SAMPLE = 'SAMPLE'
RULE = 'rule'
TOKEN = 'token'
NAMED = 'named'
PATTERN = 'pattern'
MIN_LENGTH = '_min_length'
OPTIONAL_VOID = 'void'
EMPTY_CLOSURE = 'empty_closure'
EMPTY_LIST = 'empty_list'


DEFAULT_MIN_LENGTH_BY_RULE_AND_FIELD = {
    ('either_types', 'type_names'): 2,
    ('either_color_types', 'type_names'): 2,
    ('either_orientation_types', 'type_names'): 2,
    ('either_side_types', 'type_names'): 2,

    ('multiple_args_equal_comparison', 'equal_comp_args'): 2,

    ('setup_and', 'and_args'): 2,
    ('setup_or', 'or_args'): 2,

    ('super_predicate_and', 'and_args'): 2,
    ('super_predicate_or', 'or_args'): 2,

    ('then', 'then_funcs'): 3,

    ('terminal_and', 'and_args'): 2,
    ('terminal_or', 'or_args'): 2,

    ('scoring_multi_expr', 'expr'): 2,
    ('scoring_equals_comp', 'expr'): 2,
}


HARDCODED_RULES = {
    EMPTY_CLOSURE: {
        TYPE_POSTERIOR: {RULE: 0.0, TOKEN: 1.0},
        TOKEN_POSTERIOR: {EMPTY_LIST: 1.0},
        SAMPLERS: {EMPTY_LIST: sample_empty_list},
        PRODUCTION: ((TOKEN, []),)
    },
    # COLOR: {
    #     TYPE_POSTERIOR: {RULE: 0.0, TOKEN: 1.0},
    #     TOKEN_POSTERIOR: {c: 1.0 / len(COLORS) for c in COLORS},
    #     PRODUCTION: ((TOKEN, SAMPLE),),
    # },
    # ORIENTATION: {
    #     TYPE_POSTERIOR: {RULE: 0.0, TOKEN: 1.0},
    #     TOKEN_POSTERIOR: {c: 1.0 / len(ORIENTATIONS) for c in ORIENTATIONS},
    #     PRODUCTION: ((TOKEN, SAMPLE),),
    # },
    # SIDE: {
    #     TYPE_POSTERIOR: {RULE: 0.0, TOKEN: 1.0},
    #     TOKEN_POSTERIOR: {c: 1.0 / len(SIDES) for c in SIDES},
    #     PRODUCTION: ((TOKEN, SAMPLE),),
    # },
}

class ASTSampler:
    """
    I have a prior parser into a dict format of sort.
    If I start sampling from the `start` token, that should get me somewhere.
    What I need to figure out is how to combine the information from the
    `ASTCounter` into the prior -- which of these formats is more conducive to sampling from?
    Can I transform the `ASTCounter` into a format more like the dict in which I have the prior in?

    There are a few pieces of information I want to get from the Counter:
    - For optional fields, how likely they are to exist
    - For list/variable length fields, a distribution over lengths (+ some prior?)
    - For mandatory fields, how likely the different productions are to exist
    (agin, + a prior over the rest of the productions)

    I have some thoughts in the comment above and the slack messages about how to
    handle this for the terminal productions. I need to figure out how to combine
    the information for the rule-based productions, though.
    """
    def __init__(self, grammar_parser: tatsu.grammars.Grammar,
                 ast_counter: ASTRuleValueCounter,
                 max_sample_depth: typing.Optional[int] = None,
                 max_sample_nodes: typing.Optional[int] = None,
                 pattern_rule_options: typing.Dict[str, typing.Dict[typing.Tuple[str, str], typing.Callable]] = DEFAULT_PATTERN_RULE_OPTIONS_BY_RULE,  # type: ignore
                 rule_field_value_types: typing.Dict[typing.Tuple[str, str], typing.Union[str, typing.Tuple[str]]] = SPECIAL_RULE_FIELD_VALUE_TYPES,  # type: ignore
                 pattern_type_mappings: typing.Dict[str, str] = PATTERN_TYPE_MAPPINGS,
                 local_context_propagating_rules: typing.Set[str] = LOCAL_CONTEXT_PROPAGATING_RULES,
                 omit_rules: typing.Optional[typing.Sequence[str]] = None,
                 omit_tokens: typing.Optional[typing.Sequence[str]] = None,
                 prior_rule_count: int = PRIOR_COUNT, prior_token_count: int = PRIOR_COUNT,
                 length_prior: typing.Dict[int, int] = LENGTH_PRIOR,
                 min_length_by_rule_and_field: typing.Optional[typing.Dict[typing.Tuple[str, str], int]] = DEFAULT_MIN_LENGTH_BY_RULE_AND_FIELD,  # type: ignore
                 hardcoded_rules: typing.Dict[str, dict] = HARDCODED_RULES,
                 single_literal_rules: typing.Set[str] = SINGLE_LITERAL_RULES,
                 verbose: bool = False,
                 rng: typing.Optional[np.random.Generator] = None,
                 seed: int = DEFAULT_RANDOM_SEED):

        self.grammar_parser = grammar_parser
        self.ast_counter = ast_counter
        if max_sample_depth is None:
            max_sample_depth = float('Inf')  # type: ignore
        self.max_sample_depth = max_sample_depth
        if max_sample_nodes is None:
            max_sample_nodes = float('Inf')  # type: ignore
        self.max_sample_nodes = max_sample_nodes

        self.pattern_rule_options = pattern_rule_options
        self.rule_field_value_types = rule_field_value_types
        self.pattern_type_mappings = pattern_type_mappings
        self.local_context_propagating_rules = local_context_propagating_rules

        if omit_rules is None:
            omit_rules = []
        self.omit_rules = set(omit_rules)
        if omit_tokens is None:
            omit_tokens = []
        self.omit_tokens = set(omit_tokens)

        self.prior_rule_count = prior_rule_count
        self.prior_token_count = prior_token_count
        self.length_prior = length_prior

        if min_length_by_rule_and_field is None:
            min_length_by_rule_and_field = {}

        self.min_length_by_rule_and_field = min_length_by_rule_and_field
        self.verbose = verbose

        if rng is None:
            rng = np.random.default_rng(seed)  # type: ignore
        self.rng = rng

        self.single_literal_rules = single_literal_rules
        self.rules = {k: v for k, v in hardcoded_rules.items()}
        self.value_patterns = dict(
            any=re.compile(re.escape('(any)')),
            total_time=re.compile(re.escape('(total-time)')),
            total_score=re.compile(re.escape('(total-score)'))
        )
        self.all_rules = set([rule.name for rule in self.grammar_parser.rules])
        self.all_rules.update(self.rules.keys())
        self.parse_prior_to_posterior()

        self.sample_parseinfo_index = 0

    def _update_prior_to_posterior(self, rule_name: str, field_name: str, field_prior: ContextDict):
        rule_counter = self.ast_counter.counters[rule_name]

        if OPTIONS not in field_prior:
            if field_name in rule_counter:
                print(f'No options for {rule_name}.{field_name} with counted data: {rule_counter.field_name}')  # type: ignore
            else:
                print(f'No options for {rule_name}.{field_name} with counted data: {rule_counter}')

            return

        options = field_prior[OPTIONS]
        field_counter = rule_counter[field_name] if field_name in rule_counter else None

        if MIN_LENGTH in field_prior:
            self._create_length_posterior(rule_name, field_name, field_prior, field_counter)

        if isinstance(options, str):
            options = [options]

        if isinstance(options, list):
            # If it's a list of length 1, it could be something that expands to a token, so check that
            if len(options) == 1 and options[0] in self.pattern_rule_options:
                self._create_value_posterior(rule_name, field_name, field_prior, options[0], field_counter)

            # Edge case to handle the new way I write down some of the token expanding rules
            # If there are rule counts, it's a hybrid, and we'll fall through from the rule posterior
            elif (rule_name, field_name) in self.rule_field_value_types and ((field_counter is None) or (len(field_counter.rule_counts) == 0)):
                value_type = self.rule_field_value_types[(rule_name, field_name)]
                if not isinstance(value_type, str):
                    raise ValueError(f'Unrecognized value type for {rule_name}.{field_name}: {value_type} (expected a string)')
                self._create_value_posterior(rule_name, field_name, field_prior,
                                             value_type, field_counter)
            # It's a list, which at this point means it's a list of optional expansion rules
            else:
                self._create_rule_posterior(rule_name, field_name, field_prior, options, field_counter)

        elif not isinstance(options, str):
            if self.verbose: print(f'Unrecognized options type for {rule_name}.{field_name}: {options}')

        # TODO: should these values come from the prior, likelihood, or posterior? Or be fixed?
        rule_counts = sum(field_prior[RULE_POSTERIOR].values()) if RULE_POSTERIOR in field_prior else 0
        token_counts = sum(field_prior[TOKEN_POSTERIOR].values()) if TOKEN_POSTERIOR in field_prior else 0
        type_posterior = {RULE: rule_counts, TOKEN: token_counts}
        field_prior[TYPE_POSTERIOR] = self._normalize_posterior_dict(type_posterior)  # type: ignore

        # Normalize the rule and token posteriors
        if RULE_POSTERIOR in field_prior:
            field_prior[RULE_POSTERIOR] = self._normalize_posterior_dict(field_prior[RULE_POSTERIOR], omit_keys=self.omit_rules)
        if TOKEN_POSTERIOR in field_prior:
            field_prior[TOKEN_POSTERIOR] = self._normalize_posterior_dict(field_prior[TOKEN_POSTERIOR], omit_keys=self.omit_tokens)

    def _normalize_posterior_dict(self, posterior_dict: typing.Dict[str, float], omit_keys: typing.Optional[typing.Set[str]] = None):
        if omit_keys is None:
            omit_keys = set()

        total_counts = sum([count for key, count in posterior_dict.items() if key not in omit_keys])  # type: ignore
        if total_counts == 0:
            total_counts = 1

        return {key: count / total_counts for key, count in posterior_dict.items() if key not in omit_keys}  # type: ignore

    def _create_length_posterior(self, rule_name: str, field_name: str,
        field_prior: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]],
        field_counter: typing.Optional[RuleKeyValueCounter]):

        min_length = self.min_length_by_rule_and_field.get((rule_name, field_name), typing.cast(int, field_prior[MIN_LENGTH]))
        field_prior[MIN_LENGTH] = min_length  # type: ignore
        length_posterior = Counter({k: v for k, v in self.length_prior.items() if k >= min_length})

        if field_counter is not None:
            if len(field_counter.length_counts) == 0:
                logger.warning(f'No length counts for {rule_name}.{field_name} which has a min length, filling in with 1')

            length_posterior.update(field_counter.length_counts)
            total_lengths = sum(field_counter.length_counts)
            total_obs = sum(field_counter.rule_counts.values()) + sum(field_counter.value_counts.values())

            if total_lengths > total_obs:
                raise ValueError(f'Length counts for {rule_name}.{field_name} are too high: {total_lengths} > {total_obs}')

            elif total_lengths < total_obs:
                length_posterior[1] += total_obs - total_lengths

        field_prior[LENGTH_POSTERIOR] = self._normalize_posterior_dict(length_posterior)  # type: ignore

    def _create_value_posterior(self, rule_name: str, field_name: str,
        field_prior: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]],
        value_type: str, field_counter: typing.Optional[RuleKeyValueCounter], rule_hybrird: bool = False):

        field_default = self.pattern_rule_options[value_type][(rule_name, field_name)]
        if TOKEN_POSTERIOR not in field_prior:
            field_prior[TOKEN_POSTERIOR] = defaultdict(int)

        pattern_type = value_type
        if pattern_type in self.pattern_type_mappings:
            pattern_type = self.pattern_type_mappings[pattern_type]

        value_pattern = self.value_patterns[pattern_type] if pattern_type in self.value_patterns else None

        if isinstance(field_default, list):
            if not isinstance(field_prior[TOKEN_POSTERIOR], dict):
                raise ValueError(f'Prior for {rule_name}.{field_name} is not a dict')

            field_prior[TOKEN_POSTERIOR].update({value: self.prior_token_count for value in field_default})  # type: ignore

            if field_counter is not None:
                if len(field_counter.rule_counts) > 0 and not rule_hybrird:
                    raise ValueError(f'{rule_name}.{field_name} has counted rules, which should not exist')

                for value, count in field_counter.value_counts.items():
                    if value_pattern is None or value_pattern.match(value) is not None:
                        field_prior[TOKEN_POSTERIOR][value] += count  # type: ignore

        elif hasattr(field_default, '__call__'):
            count = self.prior_token_count

            if value_pattern is not None and field_counter is not None:
                valid_values = [v for v in field_counter.value_counts if value_pattern.match(v) is not None]
                count += sum(field_counter.value_counts[value] for value in valid_values)

            field_prior[TOKEN_POSTERIOR][value_type] = count  # type: ignore
            if SAMPLERS not in field_prior:
                field_prior[SAMPLERS] = {}

            field_sampler = field_default

            if hasattr(field_default, 'factory') and field_default.factory:
                field_sampler = field_default(field_counter, self.prior_token_count)

            field_prior[SAMPLERS][value_type] = field_sampler  # type: ignore

        else:
            raise ValueError(f'Unknown field_default type: {field_default}')

    def _create_rule_posterior(self, rule_name: str, field_name: str,
        field_prior: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]],
        options: typing.Sequence[str], field_counter: typing.Optional[RuleKeyValueCounter]):

        field_prior[RULE_POSTERIOR] = {value: self.prior_token_count for value in options}

        if field_counter is not None:
            if (rule_name, field_name) in self.rule_field_value_types:
                value_type = self.rule_field_value_types[(rule_name, field_name)]
                if isinstance(value_type, str):
                    if value_type in field_prior[RULE_POSTERIOR]:
                        del field_prior[RULE_POSTERIOR][value_type]  # type: ignore
                    self._create_value_posterior(rule_name, field_name, field_prior, value_type, field_counter, rule_hybrird=True)
                else:
                    for vt in value_type:
                        if vt in field_prior[RULE_POSTERIOR]:
                            del field_prior[RULE_POSTERIOR][vt]  # type: ignore
                        self._create_value_posterior(rule_name, field_name, field_prior, vt, field_counter, rule_hybrird=True)

            elif len(field_counter.value_counts) > 0:
                raise ValueError(f'{rule_name}.{field_name} has counted values but should only have rules or have a special type')

            for counted_rule, count in field_counter.rule_counts.items():
                if counted_rule not in field_prior[RULE_POSTERIOR]:
                    raise ValueError(f'{rule_name}.{field_name} has counted rule {counted_rule} which is not in the prior {options}')

                field_prior[RULE_POSTERIOR][counted_rule] += count  # type: ignore
        else:
            if self.verbose: print(f'No counted data for {rule_name}.{field_name}')

        if len(field_prior[RULE_POSTERIOR]) == 0:
            del field_prior[RULE_POSTERIOR]

    def parse_prior_to_posterior(self):
        for rule in self.grammar_parser.rules:
            children = rule.children()
            if len(children) > 1:
                print(f'Encountered rule with multiple children: {rule.name}')
                continue

            child = children[0]
            rule_name = rule.name
            rule_prior = self.parse_rule_prior(child)

            # In case it's one of the hard-coded rules
            if rule_name in self.rules:
                continue

            # Special cases
            if rule_name in ('preferences', 'pref_forall_prefs'):
                if not isinstance(rule_prior, (list, tuple)):
                    raise ValueError(f'Prior for {rule_name} is not a list or tuple')

                rule_prior = rule_prior[0]  # type: ignore

            if rule_name == START:
                pass

            elif isinstance(rule_prior, dict):
                production = None

                for field_name, field_prior in rule_prior.items():
                    production = None
                    if field_name == PRODUCTION:
                        continue

                    if field_name == PATTERN and isinstance(field_prior, str):
                        production = [(PATTERN, field_prior)]
                        self.value_patterns[rule_name] = re.compile(field_prior)
                        continue

                    if not isinstance(field_prior, dict):
                        print(f'Encountered non-dict prior for {rule_name}.{field_name}: {field_prior}')
                        continue

                    self._update_prior_to_posterior(rule_name, field_name, field_prior)

                if PRODUCTION not in rule_prior:
                    if production is None:
                        production = [(NAMED, key) for key in rule_prior.keys()]

                    rule_prior[PRODUCTION] = production

            elif isinstance(rule_prior, list):
                # The sections that optionally exist
                if OPTIONAL_VOID in rule_prior:
                    section_name = rule_name.replace('_def', '')
                    if section_name not in self.ast_counter.section_counts:
                        raise ValueError(f'{rule_name} has no section counts')

                    section_prob = self.ast_counter.section_counts[section_name] / max(self.ast_counter.section_counts.values())
                    child = [x for x in rule_prior if x is not None][0]
                    child[PRODUCTION_PROBABILITY] = section_prob
                    rule_prior = child

                else:
                    if rule_name == 'predicate_name':
                        rule_prior = dict(rule_posterior={r: 1.0 / len(rule_prior) for r in rule_prior}, production=[('rule', SAMPLE)])

                    else:
                        rule_prior = dict(token_posterior={r: 1.0 / len(rule_prior) for r in rule_prior}, production=[('token', SAMPLE)])


            elif isinstance(rule_prior, str):
                # This is a rule that expands only to a single other rule
                if rule_prior in self.all_rules and not rule_name in self.single_literal_rules:
                    rule_prior = dict(rule_posterior={rule_prior: 1}, production=[(RULE, rule_prior)])

                # This is a rule that expands directly to a token
                else:
                    token = rule_prior
                    rule_prior = dict(token_posterior={token: 1}, production=[(TOKEN, token)])
                    self.value_patterns[rule_name] = re.compile(re.escape(token))
                    if self.verbose: print(f'String token rule for {rule_name}: {rule_prior}')

            else:
                raise ValueError(f'Encountered rule with unknown prior or no special case: {rule.name}\n{rule_prior}')

            self.rules[rule.name] = rule_prior  # type: ignore

    def parse_rule_prior(self, rule: tatsu.grammars.Node) -> typing.Union[None, str, ContextDict, typing.List]:
        if isinstance(rule, grammars.EOF):
            return EOF

        if isinstance(rule, grammars.Token):
            return rule.token

        if isinstance(rule, grammars.Pattern):
            return dict(pattern=rule.pattern)

        if isinstance(rule, grammars.Sequence):
            rule_dict = {}
            sequence = []
            for child in rule.children():
                parsed_child = self.parse_rule_prior(child)
                if isinstance(parsed_child, str):
                    if isinstance(child, grammars.RuleRef):
                        sequence.append((RULE, parsed_child))
                    else:
                        sequence.append((TOKEN, parsed_child))

                elif isinstance(parsed_child, dict):
                    rule_dict.update(parsed_child)
                    if len(parsed_child.keys()) > 1:
                        print(f'Encountered child rule parsing to dict with multiple keys: {rule.name}: {parsed_child}')  # type: ignore
                    else:
                        sequence.append((NAMED, list(parsed_child.keys())[0]))

                else:
                    print(f'Encountered child rule parsing to unknown type: {rule.name}: {parsed_child}')  # type: ignore

            rule_dict[PRODUCTION] = sequence
            return rule_dict

        if isinstance(rule, grammars.Named):
            children = rule.children()
            if len(children) > 1:
                raise ValueError(f'Named rule has more than one child: {rule}')

            child_prior = self.parse_rule_prior(children[0])
            # if isinstance(child_prior, str) and child_prior not in self.all_rules:
            #     return child_prior

            return {rule.name: dict(options=child_prior)}

        if isinstance(rule, grammars.Group):
            if len(rule.children()) == 1:
                return self.parse_rule_prior(rule.children()[0])
            else:
                return [self.parse_rule_prior(child) for child in rule.children()]

        if isinstance(rule, grammars.Choice):
            return [self.parse_rule_prior(child) for child in rule.children()]

        if isinstance(rule, grammars.Option):
            children = rule.children()
            if len(children) > 1 :
                raise ValueError(f'Option rule has more than one child: {rule}')

            return self.parse_rule_prior(children[0])

        if isinstance(rule, (grammars.PositiveClosure, grammars.Closure)):
            d = {MIN_LENGTH: 1 if isinstance(rule, grammars.PositiveClosure) else 0}
            if len(rule.children()) == 1:
                child_value = self.parse_rule_prior(rule.children()[0])
                if not isinstance(child_value, dict):
                    print(f'Encoutered positive closure with unexpected value type: {child_value}')

                child_value[next(iter(child_value))].update(d)  # type: ignore
                d = child_value
            else:
                print(f'Encoutered positive closure with multiple children: {rule}')

            return d

        if isinstance(rule, grammars.RuleRef):
            return rule.name

        if isinstance(rule, grammars.Void):
            return OPTIONAL_VOID

        if isinstance(rule, grammars.EmptyClosure):
            return EMPTY_CLOSURE

        raise ValueError(f'Encountered unknown rule type: {type(rule)}: {rule}')

    def _sample_named(self,
        sample_dict: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]],
        global_context: ContextDict,
        local_context: ContextDict):

        if TYPE_POSTERIOR not in sample_dict:
            raise ValueError(f'Missing type_posterior in sample: {sample_dict}')

        if LENGTH_POSTERIOR in sample_dict:
            length = posterior_dict_sample(global_context['rng'], sample_dict[LENGTH_POSTERIOR])  # type: ignore
            if length == 0:
                return None, None
            values, context_updates = zip(*[self._sample_single_named_value(sample_dict, global_context, local_context) for _ in range(length)])
            context_update = _combine_context_updates(context_updates)  # type: ignore
            return list(values), context_update

        else:
            return self._sample_single_named_value(sample_dict, global_context, local_context)

    def _sample_single_named_value(self, sample_dict: typing.Dict[str, typing.Union[str, typing.Sequence[str], typing.Dict[str, float]]],
        global_context: ContextDict,
        local_context: ContextDict):

        sample_type = posterior_dict_sample(global_context['rng'], sample_dict[TYPE_POSTERIOR])  # type: ignore

        if sample_type == RULE:
            if RULE_POSTERIOR not in sample_dict:
                raise ValueError(f'Missing rule_posterior in sample: {sample_dict}')

            rule = str(posterior_dict_sample(global_context['rng'], sample_dict[RULE_POSTERIOR]))  # type: ignore
            return self.sample(rule, global_context, local_context, False)

        elif sample_type == TOKEN:
            if TOKEN_POSTERIOR not in sample_dict:
                raise ValueError(f'Missing token_posterior in sample: {sample_dict}')

            token = posterior_dict_sample(global_context['rng'], sample_dict[TOKEN_POSTERIOR])  # type: ignore
            if SAMPLERS in sample_dict and token in sample_dict[SAMPLERS]:  # type: ignore
                token = sample_dict[SAMPLERS][token](global_context, local_context)     # type: ignore

            return token, None

        else:
            raise ValueError(f'Unknown sample type: {sample_type} in sample: {sample_dict}')

    def sample(self, rule: str = START,
        global_context: typing.Optional[ContextDict] = None,
        local_context: typing.Optional[ContextDict] = None,
        initial_call: bool = True,):

        if rule == START:
            self.sample_parseinfo_index = 0

        if global_context is None:
            global_context = dict(rng=self.rng)
        elif 'rng' not in global_context:
            global_context['rng'] = self.rng

        if initial_call:
            global_context['total_nodes_generated'] = 0

        global_context['total_nodes_generated'] += 1
        if global_context['total_nodes_generated'] > self.max_sample_nodes:  # type: ignore
            raise SamplingException(f'Exceeded max nodes generated: {self.max_sample_nodes}')

        if local_context is None:
            local_context = dict()
        else:
            local_context = simplified_context_deepcopy(local_context)

        if 'depth' not in local_context:
            local_context['depth'] = 0

        local_context['depth'] += 1
        if local_context['depth'] > self.max_sample_depth:  # type: ignore
            raise SamplingException(f'Exceeded max sample depth: {self.max_sample_depth}')

        rule_dict = self.rules[rule]
        production = rule_dict[PRODUCTION]
        output = []
        return_ast = False

        if PRODUCTION_PROBABILITY in rule_dict and not initial_call:
            if global_context['rng'].random() > rule_dict[PRODUCTION_PROBABILITY]:
                return None, None

        for prod_type, prod_value in production:
            if prod_type == TOKEN:
                if prod_value == EOF:
                    pass
                elif prod_value == SAMPLE:
                    output.append(posterior_dict_sample(global_context['rng'], rule_dict[TOKEN_POSTERIOR]))
                else:
                    output.append(prod_value)

            elif prod_type == RULE:
                value, context_update = self.sample(prod_value, global_context, local_context, False)
                if context_update is not None:
                    local_context.update(context_update)
                if value is not None:
                    output.append(value)

            elif prod_type == NAMED:
                return_ast = True
                value, context_update = self._sample_named(rule_dict[prod_value],  global_context, local_context)
                if context_update is not None:
                    local_context.update(context_update)
                output.append({prod_value: value})

        if len(output) == 0:
            print(f'Encountered empty production for {rule}: {production}')
            return None, None

        if return_ast:
            out_dict = _combine_context_updates(output)
            out_dict['parseinfo'] = tatsu.infos.ParseInfo(None, rule, self.sample_parseinfo_index, self.sample_parseinfo_index, self.sample_parseinfo_index, self.sample_parseinfo_index)
            self.sample_parseinfo_index += 1
            output = tatsu.ast.AST(out_dict)

        elif len(output) == 1:
            output = output[0]

        else:
            output = tuple(output)

        if initial_call:
            global_context.pop('total_nodes_generated')

        if rule in self.local_context_propagating_rules:
            return output, local_context

        elif rule == START:
            return output

        return output, None

    def sample_until_success(self, rule: str = START,
        global_context: typing.Optional[ContextDict] = None,
        local_context: typing.Optional[ContextDict] = None,
        max_attempts: int = 1000):

        for _ in range(max_attempts):
            try:
                return self.sample(rule, global_context, local_context)
            except SamplingException:
                pass

        raise SamplingException(f'Failed to sample {rule} after {max_attempts} attempts')


SECTION_SAMPLE_WEIGHTS = {
    'uniform': {key: 1.0 for key in SECTION_KEYS},
    '2_to_1': {key: 2.0 if key == PREFERENCES else 1.0 for key in SECTION_KEYS},
}


def uniform_depth_weight(depths: np.ndarray) -> np.ndarray:
    return np.ones_like(depths)


def quadratic_depth_weight(depths: np.ndarray) -> np.ndarray:
    min_depth, max_depth = np.min(depths), np.max(depths)
    return - (depths - min_depth) * (depths - max_depth) + 1


DEPTH_WEIGHT_FUNCTIONS = {
    'uniform': uniform_depth_weight,
    'quadratic': quadratic_depth_weight,
}


def _regrowth_sampler_state_wrapper(func: typing.Callable) -> typing.Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        source_ast = self.source_ast
        parent_mapping = self.parent_mapping
        node_keys = self.node_keys
        node_keys_by_section = self.node_keys_by_section

        try:
            return func(self, *args, **kwargs)

        finally:
            self.source_ast = source_ast
            self.parent_mapping = parent_mapping
            self.node_keys = node_keys
            self.node_keys_by_section = node_keys_by_section

    return wrapper


def _update_game_id(ast: typing.Union[tuple, tatsu.ast.AST], sample_index: int, suffix: typing.Optional[typing.Any] = None) -> tuple:
    new_game_name = f'{ast[1].game_name}-{sample_index}{"-" + str(suffix) if suffix else ""}'  # type: ignore
    replace_child(ast[1], ['game_name'], new_game_name)  # type: ignore
    return ast  # type: ignore


class RegrowthSampler(ASTParentMapper):
    depth_parser: ASTDepthParser
    depth_weight_function: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]
    fix_contexts: bool
    node_keys: typing.List[tatsu.infos.ParseInfo]
    node_keys_by_section: typing.Dict[str, typing.List[tatsu.infos.ParseInfo]]
    original_game_id: str
    parent_mapping: ASTParentMapping
    rng: np.random.Generator
    samplers: typing.Dict[str, ASTSampler]
    section_sample_weights: typing.Optional[typing.Dict[str, float]]
    seed: int
    source_ast: typing.Union[tuple, tatsu.ast.AST]

    def __init__(self, sampler: typing.Union[ASTSampler, typing.Dict[str, ASTSampler]],
                 section_sample_weights: typing.Optional[typing.Dict[str, float]] = None,
                 depth_weight_function: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
                 fix_contexts: bool = False, seed: int = 0, rng: typing.Optional[np.random.Generator] = None):

        if isinstance(sampler, ASTSampler):
            sampler = dict(default=sampler)
        self.samplers = sampler
        self.sampler_keys = list(self.samplers.keys())
        self.example_sampler = self.samplers[self.sampler_keys[0]]

        super().__init__(local_context_propagating_rules=self.example_sampler.local_context_propagating_rules)

        self.section_sample_weights = section_sample_weights
        if self.section_sample_weights is not None:
            section_sample_weights_sum = sum(self.section_sample_weights.values())
            self.section_sample_weights = {key: value / section_sample_weights_sum for key, value in self.section_sample_weights.items()}
        self.depth_weight_function = depth_weight_function
        self.fix_contexts = fix_contexts
        self.seed = seed

        if rng is None:
            rng = np.random.default_rng(seed)
        self.rng = rng

        if self.fix_contexts:
            self.context_fixer = ast_context_fixer.ASTContextFixer(self.example_sampler, self.rng, strict=False)

        self.parent_mapping = dict()
        self.depth_parser = ASTDepthParser()
        self.source_ast = None  # type: ignore
        self.node_keys = []
        self.node_keys_by_section = {}

    def set_source_ast(self, source_ast: typing.Union[tuple, tatsu.ast.AST]):
        self.source_ast = source_ast
        self.parent_mapping = {}
        self(source_ast)
        self.node_keys = list(
            key for key, node_info
            in self.parent_mapping.items()
            if isinstance(node_info.parent, tatsu.ast.AST)
        )

        if self.section_sample_weights is not None:
            self.node_keys_by_section = defaultdict(list)
            for key, node_info in self.parent_mapping.items():
                if isinstance(node_info.parent, tatsu.ast.AST) and node_info.section is not None:
                    self.node_keys_by_section[node_info.section].append(key)

    def _parse_current_node(self, ast, **kwargs):
        if ast.parseinfo.rule == 'game_def':
            self.original_game_id = ast.game_name

    def _sample_node_to_update(self, rng: np.random.Generator):
        if self.section_sample_weights is not None:
            game_sections = [section for section in self.section_sample_weights.keys() if len(self.node_keys_by_section[section]) > 0]
            game_section_weights = np.array([self.section_sample_weights[section] for section in game_sections])
            game_section_weights = game_section_weights / np.sum(game_section_weights)
            section = rng.choice(game_sections, p=game_section_weights)
            node_key_list = self.node_keys_by_section[section]

        else:
            node_key_list = self.node_keys

        if self.depth_weight_function is not None:
            node_depths = [self.parent_mapping[node_key].depth for node_key in node_key_list]
            node_weights_by_depth = self.depth_weight_function(np.array(node_depths))
            node_weights = node_weights_by_depth / np.sum(node_weights_by_depth)
            node_index = rng.choice(len(node_key_list), p=node_weights)

        else:
            node_index = rng.choice(len(node_key_list))

        node_key = self.node_keys[node_index]
        return self.parent_mapping[node_key]

    def _find_node_depth(self, node: tatsu.ast.AST):
        node_info = self.parent_mapping[self._ast_key(node)]
        depth = 1
        while not (isinstance(node_info[1], tuple) and node_info[1][0] == '(define'):
            node_info = self.parent_mapping[self._ast_key(node_info[1])]
            depth += 1

        return depth

    @_regrowth_sampler_state_wrapper
    def sample(self, sample_index: int, n_regrowths: int = 1,
               external_global_context: typing.Optional[ContextDict] = None,
               external_local_context: typing.Optional[ContextDict] = None, update_game_id: bool = True,
               rng: typing.Optional[np.random.Generator] = None, node_key_to_regrow: typing.Optional[typing.Hashable] = None) -> typing.Union[tatsu.ast.AST, tuple]:

        if rng is None:
            rng = self.rng

        new_source = typing.cast(tuple, deepcopy_ast(self.source_ast))

        for _ in range(n_regrowths):
            self.set_source_ast(new_source)

            if node_key_to_regrow is not None:
                if node_key_to_regrow not in self.parent_mapping:
                    raise ValueError(f'Node key to regrow not found: {node_key_to_regrow}')

                node, parent, selector, node_depth, section, global_context, local_context = self.parent_mapping[node_key_to_regrow]

            else:
                node, parent, selector, node_depth, section, global_context, local_context = self._sample_node_to_update(rng)  # type: ignore

            if section is None: section = ''

            if external_global_context is not None:
                global_context = global_context.copy()
                global_context.update(external_global_context)

            if external_local_context is not None:
                local_context = local_context.copy()
                local_context.update(external_local_context)

            global_context['original_game_id'] = self.original_game_id
            global_context['rng'] = rng

            sampler_key = rng.choice(self.sampler_keys)
            sampler = self.samplers[sampler_key]

            try:
                new_node = sampler.sample(node.parseinfo.rule, global_context, local_context)[0]  # type: ignore

                if update_game_id:
                    regrwoth_depth = self.depth_parser(node)
                    new_source = _update_game_id(new_source, sample_index, f'nd-{node_depth}-rd-{regrwoth_depth}-rs-{section.replace("(:", "")}-sk-{sampler_key}-nr-{n_regrowths}')

                replace_child(parent, selector, new_node)

                if self.fix_contexts:
                    self.context_fixer.fix_contexts(new_source, original_child=node, crossover_child=new_node)  # type: ignore

            except IndexError as e:
                logger.error(f'Caught IndexError in RegrowthSampler.sample:')
                logger.error(f'  node: {node}')
                logger.error(f'  parent: {parent}')
                logger.error(f'  selector: {selector}')
                logger.error(f'  len(parent[selector[0]]): {len(parent[selector[0]])}')  # type: ignore
                logger.error(f'  node_depth: {node_depth}')
                logger.error(f'  section: {section}')
                logger.error(f'  global_context: {global_context}')
                logger.error(f'  local_context: {local_context}')
                logger.error(f'  sampler_key: {sampler_key}')
                raise e


        return new_source


def parse_or_load_counter(args: argparse.Namespace, grammar_parser: typing.Optional[tatsu.grammars.Grammar] = None):
    if args.parse_counter:
        if grammar_parser is None:
            raise ValueError('Grammar parser must be provided if parsing counter')

        counter = ASTRuleValueCounter()

        for test_file in args.test_files:
            for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, not args.dont_tqdm, args.relative_path):
                counter(ast)

        with open(args.counter_output_path, 'wb') as out_file:
            pickle.dump(counter, out_file)

    else:
        with open(args.counter_output_path, 'rb') as pickle_file:
            counter = pickle.load(pickle_file)

    return counter


def test_and_stringify_ast_sample(ast, args: argparse.Namespace, grammar_parser: tatsu.grammars.Grammar):
    first_print_out = ''

    try:
        if args.print_samples:
            ast_printer.BUFFER = None
            ast_printer.pretty_print_ast(ast)
            print()

        if args.validate_samples or args.save_samples:
            first_print_out = ast_printer.ast_to_string(ast, line_delimiter='\n')

        if args.validate_samples:
            second_ast = grammar_parser.parse(first_print_out)
            second_print_out = ast_printer.ast_to_string(second_ast, line_delimiter='\n')  # type: ignore

            if first_print_out != second_print_out:
                print('Mismatch found')

    except (tatsu.exceptions.FailedToken, tatsu.exceptions.FailedParse) as e:
        print(f'Parse failed: at position {e.pos} expected "{e.item}" :')
        if len(first_print_out) > e.pos:
            print(first_print_out[e.pos:])

    return first_print_out


def _generate_mle_samples(args: argparse.Namespace, samplers: typing.Union[typing.Dict[str, ASTSampler], typing.List[typing.Dict[str, ASTSampler]]], grammar_parser: tatsu.grammars.Grammar):
    if args.sample_parallel:
        raise ValueError('MLE sampling not supported with parallel sampling')

    sample_iter = range(args.num_samples)
    if args.sample_tqdm:
        sample_iter = tqdm.tqdm(sample_iter, desc='Samples')

    rng = np.random.default_rng(args.random_seed)

    for sample_id in sample_iter:
        sampler_key = rng.choice(list(samplers.keys()))  # type: ignore
        sampler = samplers[sampler_key]
        generated_sample = False
        while not generated_sample:
            try:
                sample_ast = sampler.sample(global_context=dict(sample_id=sample_id))
                sample_str = test_and_stringify_ast_sample(sample_ast, args, grammar_parser)
                generated_sample = True
                yield sample_str + '\n\n'

            # except ValueError as e:
            #     print(f'ValueError while sampling, repeating: {e}')
            except SamplingException as e:
                print(f'SamplingException while sampling, repeating: {e}')


def regrow_single_sample(regrowth_sampler: RegrowthSampler, sample_index: int, grammar_parser: tatsu.grammars.Grammar, args: argparse.Namespace) -> typing.Tuple[str, str, int]:
    n_regrowths = regrowth_sampler.rng.integers(args.min_n_regrowths, args.max_n_regrowths + 1)
    sampling_exception_count = 0
    while True:
        try:
            sample_ast = regrowth_sampler.sample(sample_index, n_regrowths=n_regrowths)
            sample_str = test_and_stringify_ast_sample(sample_ast, args, grammar_parser)
            sample_hash = fixed_hash(sample_str[sample_str.find('(:domain'):])
            return sample_str + '\n\n', sample_hash, sampling_exception_count

        except RecursionError:
            if args.verbose: print('Recursion error, skipping sample')

        except SamplingException:
            sampling_exception_count += 1
            if args.verbose: print('Sampling exception, skipping sample')

        except Exception as e:
            # I think something with multiprocessing causes this to sometime happen
            exception_name = type(e).__name__
            if exception_name == 'SamplingException':
                sampling_exception_count += 1
                if args.verbose: print(f'Sampling exception, skipping sample: {e}')

            else:
                # if args.verbose: print(f'Exception of type {type(e)} while sampling, skipping sample: {e}')
                raise e


def _process_index(n_workers: int):
    identity = multiprocessing.current_process()._identity
    if identity is None or len(identity) == 0:
        return 0

    return (identity[0] - 1) % n_workers


def regrow_sample_parallel(regrowth_samplers: typing.List[RegrowthSampler], sample_index: int, grammar_parsers: typing.List[tatsu.grammars.Grammar], args: argparse.Namespace):
    process_index = _process_index(args.parallel_n_workers)
    return regrow_single_sample(regrowth_samplers[process_index], sample_index, grammar_parsers[process_index], args)


def regrow_sample_parallel_map_wrapper(args):
    return regrow_sample_parallel(*args)



def _generate_regrowth_samples(args: argparse.Namespace, samplers: typing.Union[typing.Dict[str, ASTSampler], typing.List[typing.Dict[str, ASTSampler]]],
                               grammar_parser: typing.Union[tatsu.grammars.Grammar, typing.List[tatsu.grammars.Grammar]]):
    section_sample_weights = None
    if args.section_sample_weights_key is not None:
        section_sample_weights = SECTION_SAMPLE_WEIGHTS[args.section_sample_weights_key]

    depth_weight_function = None
    if args.depth_weight_function_key is not None:
        depth_weight_function = DEPTH_WEIGHT_FUNCTIONS[args.depth_weight_function_key]

    gp = grammar_parser if not isinstance(grammar_parser, list) else grammar_parser[0]
    real_games = [sample_ast for test_file in args.test_files for sample_ast
                  in cached_load_and_parse_games_from_file(test_file, gp, not args.dont_tqdm)]

    if args.regrowth_end_index == -1:
        args.regrowth_end_index = len(real_games)

    else:
        args.regrowth_end_index = min(args.regrowth_end_index, len(real_games))

    game_iter = iter(real_games[args.regrowth_start_index:args.regrowth_end_index])
    if args.sample_tqdm:
        game_iter = tqdm.tqdm(game_iter, desc=f'Game #', total=args.regrowth_end_index - args.regrowth_start_index)

    if args.sample_parallel:
        regrowth_samplers = [RegrowthSampler(samplers[worker_id], # type: ignore
                                             section_sample_weights, depth_weight_function,
                                             fix_contexts=args.fix_contexts, seed=args.random_seed + worker_id)
                             for worker_id in range(args.parallel_n_workers)]
        if not isinstance(grammar_parser, list):
            grammar_parser = [copy.deepcopy(grammar_parser) for _ in range(args.parallel_n_workers)]

        game_start = '(game '
        single_chunk_total = args.parallel_chunksize * args.parallel_n_workers

        logger.debug(f'Parallel regrowth sampling with {args.parallel_n_workers} workers and max tasks per child {args.parallel_maxtasksperchild}')

        with mpp.Pool(args.parallel_n_workers, maxtasksperchild=args.parallel_maxtasksperchild) as pool:
            for real_game in game_iter:
                for regrowth_sampler in regrowth_samplers:
                    regrowth_sampler.set_source_ast(real_game)   # type: ignore

                real_game_str = ast_printer.ast_to_string(real_game, line_delimiter='\n')  # type: ignore
                sample_hashes = set([fixed_hash(real_game_str[real_game_str.find('(:domain'):])])
                sample_index = 0
                n_maps = 0
                total_sampling_exception_count = 0

                while len(sample_hashes) < args.num_samples + 1:
                    # This is required because the changes to the rng state happen in the pickled copies in the worker processes
                    # and don't get propagated back here -- so we have to provide a different state for each map call
                    for worker_id, regrowth_sampler in enumerate(regrowth_samplers):
                        regrowth_sampler.rng = np.random.default_rng(args.random_seed + worker_id + sample_index)  # type: ignore

                    param_iterator = zip(
                        itertools.repeat(regrowth_samplers, single_chunk_total),
                        itertools.count(single_chunk_total * n_maps),
                        itertools.repeat(grammar_parser, single_chunk_total),
                        itertools.repeat(args, single_chunk_total),
                    )
                    samples_iter = pool.imap_unordered(regrow_sample_parallel_map_wrapper, param_iterator, chunksize=args.parallel_chunksize)

                    for sample_str, sample_hash, sampling_execption_count in samples_iter:
                        total_sampling_exception_count += sampling_execption_count

                        if sample_hash not in sample_hashes:
                            game_name_start_index = sample_str.find(game_start) + len(game_start)
                            game_name_end_index = sample_str.find(')', game_name_start_index)
                            game_name = sample_str[game_name_start_index:game_name_end_index]

                            game_id, game_index, _, game_name_suffix = game_name.split('-', 3)
                            new_game_name = '-'.join([game_id, game_index, str(sample_index), game_name_suffix])
                            sample_str = sample_str[:game_name_start_index] + new_game_name + sample_str[game_name_end_index:]

                            sample_hashes.add(sample_hash)
                            sample_index += 1

                            yield sample_str + '\n\n'

                            if sample_index >= args.num_samples:
                                break

                    n_maps += 1
                    if args.sample_tqdm:
                        game_iter.set_postfix({'Samples': sample_index, 'Sample Hashes': len(sample_hashes), 'Exceptions': total_sampling_exception_count, 'Maps': n_maps})  # type: ignore

    else:
        regrowth_sampler = RegrowthSampler(samplers, # type: ignore
                                           section_sample_weights, depth_weight_function,
                                           fix_contexts=args.fix_contexts, seed=args.random_seed)

        for real_game in game_iter:
            regrowth_sampler.set_source_ast(real_game)  # type: ignore
            real_game_str = ast_printer.ast_to_string(real_game, line_delimiter='\n')  # type: ignore
            sample_hashes = set([fixed_hash(real_game_str[real_game_str.find('(:domain'):])])

            sample_iter = range(args.num_samples)
            if args.inner_sample_tqdm:
                sample_iter = tqdm.tqdm(sample_iter, total=args.num_samples, desc='Samples')

            attempts = 0
            total_sampling_exception_count = 0
            for sample_index in sample_iter:
                new_sample_generated = False

                while not new_sample_generated:
                    attempts += 1
                    sample_str, sample_hash, sampling_execption_count = regrow_single_sample(regrowth_sampler, sample_index, grammar_parser, args)  # type: ignore
                    total_sampling_exception_count += sampling_execption_count

                    if sample_hash in sample_hashes:
                        if args.verbose: print('Regrowth generated identical games, repeating')
                    else:
                        new_sample_generated = True
                        sample_hashes.add(sample_hash)

                        yield sample_str + '\n\n'

                        if args.sample_tqdm:
                            game_iter.set_postfix({'Samples': sample_index, 'Attempts': attempts, 'Exceptions': total_sampling_exception_count})  # type: ignore


def main(args):
    # original_recursion_limit = sys.getrecursionlimit()
    # sys.setrecursionlimit(args.recursion_limit)

    args_str = '\n'.join([f'{" " * 26}{k}: {v}' for k, v in vars(args).items()])
    logger.debug(f'Shell arguments:\n{args_str}')

    grammar = open(args.grammar_file).read()
    grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(grammar))
    counter = parse_or_load_counter(args, grammar_parser)

    if args.sample_parallel:
        samplers = []
        grammar_parser = [tatsu.compile(grammar) for _ in range(args.parallel_n_workers)]
        for worker_id in range(args.parallel_n_workers):
            worker_samplers = {}
            for pc in args.prior_count:
                length_prior = {n: pc for n in LENGTH_PRIOR}
                worker_samplers[f'prior{pc}'] = ASTSampler(
                    grammar_parser[worker_id], counter,  # type: ignore
                    max_sample_depth=args.max_sample_depth,
                    max_sample_nodes=args.max_sample_nodes,
                    seed=args.random_seed + worker_id,
                    prior_rule_count=pc, prior_token_count=pc,
                    length_prior=length_prior)     # type: ignore

            samplers.append(worker_samplers)

    else:
        samplers = {}
        for pc in args.prior_count:
            length_prior = {n: pc for n in LENGTH_PRIOR}
            samplers[f'prior{pc}'] = ASTSampler(
                grammar_parser, counter,
                max_sample_depth=args.max_sample_depth,
                max_sample_nodes=args.max_sample_nodes,
                seed=args.random_seed,  # type: ignore
                prior_rule_count=pc, prior_token_count=pc,
                length_prior=length_prior)  # type: ignore


    if args.sampling_method == MLE_SAMPLING:
        sample_iter = _generate_mle_samples(args, samplers, grammar_parser)  # type: ignore
    elif args.sampling_method == REGROWTH_SAMPLING:
        sample_iter = _generate_regrowth_samples(args, samplers, grammar_parser)  # type: ignore
    else:
        raise ValueError(f'Unknown sampling method: {args.sampling_method}')

    # TODO: conceptual issue: in places where the expansion is recursive
    # (e.g. `setup`` expands to rules that all contain setup, or
    # `preference`` expand to exists/forall that contain preferences)
    # the inner rules (`setup_game_conserved` / `setup_game_optional`),
    # or `then` for preferences) are overrepreesnted, and more likely
    # to be sampled at the root of this expression than they are in the corpus

    # sys.setrecursionlimit(original_recursion_limit)

    if args.save_samples:
        logger.info(f'Saving samples to {args.samples_output_path}')
        open_method = gzip.open if args.samples_output_path.endswith('.gz') else open
        with open_method(args.samples_output_path, args.file_open_mode) as out_file:
            buffer = []
            i = 0
            for sample in sample_iter:
                buffer.append(sample)
                i += 1
                if i % args.num_samples == 0:
                    out_file.write('\n\n'.join(buffer))
                    out_file.flush()
                    buffer = []

    else:
        for ast, sample in sample_iter:
            continue

    return


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    if len(args.prior_count) == 0:
        args.prior_count.append(PRIOR_COUNT)

    main(args)
