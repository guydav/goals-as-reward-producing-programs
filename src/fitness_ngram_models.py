import argparse
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
import itertools
import os
import pickle
import re
import sys
import typing

from nltk.util import ngrams as nltk_ngrams
import numpy as np
import tatsu
import tatsu.ast

import ast_printer
import ast_parser
from ast_utils import cached_load_and_parse_games_from_file
import room_and_object_types

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
DEFAULT_N = 5
parser.add_argument('-n', '--n', type=int, default=DEFAULT_N)
DEFAULT_OUTPUT_PATH_PATTERN = './models/{model_type}_{n}_ngram_model_{today}.pkl'
parser.add_argument('-o', '--output-path', default=None)
parser.add_argument('--use-specific-objects', action='store_true')
DEFAULT_STUPID_BACKOFF_DISCOUNT = 0.4
parser.add_argument('--stupid-backoff-discount', type=float, default=DEFAULT_STUPID_BACKOFF_DISCOUNT)
DEFAULT_ZERO_LOG_PROB = -7
parser.add_argument('--zero-log-prob', type=float, default=DEFAULT_ZERO_LOG_PROB)
parser.add_argument('--from-asts', action='store_true')
parser.add_argument('--no-pad', action='store_true')
parser.add_argument('--padding', default=None, type=int)
parser.add_argument('--fit-shuffle-list-elements', action='store_true')
DEFAULT_RANDOM_SEED = 33
parser.add_argument('--random-seed', default=DEFAULT_RANDOM_SEED, type=int)


WHITESPACE_PATTERN = re.compile(r'\s+')
VARIABLE_PATTERN = re.compile(r'\?[A-Za-z0-9_]+')
PREFERENCE_NAME_PATTERN = re.compile(r'\(preference\s+([A-Za-z0-9_]+)\s+')
NUMBER_AND_DECIMAL_PATTERN = re.compile(r'\-?[0-9]+(\.[0-9]+)?')
NON_TOKEN_CHARACTERS_PATTERN = re.compile(r'(\(:)|\(|\)|( \- )')


def ngram_preprocess(game_text: str) -> typing.List[str]:
    # remove game preamble
    domain_start = game_text.find('(:domain')
    domain_end = game_text.find(')', domain_start)
    game_text = game_text[domain_end + 1:]

    # TODO: convert variables to category types
    # remove variables
    game_text = VARIABLE_PATTERN.sub('', game_text)
    # replace preference names
    for preference_name in PREFERENCE_NAME_PATTERN.findall(game_text):
        game_text = game_text.replace(preference_name, 'preferenceName')
    # remove numbers and decimals
    game_text = NUMBER_AND_DECIMAL_PATTERN.sub('number', game_text)
    # remove non-token characters
    game_text = NON_TOKEN_CHARACTERS_PATTERN.sub('', game_text)
    # replace any remaining colons with spaces
    game_text = game_text.replace(':', ' ')
    # standardize whitespace
    game_text = WHITESPACE_PATTERN.sub(' ', game_text)
    # tokenize
    tokens = game_text.strip().split()
    # convert types to categories
    tokens = [room_and_object_types.TYPES_TO_CATEGORIES.get(token, token) for token in tokens]
    return tokens


START_PAD = '<start>'
END_PAD = '<end>'
UNKNOWN_CATEGORY = '<unknown>'

def _ngrams(text: str, n: int, pad: int = 0, start_pad: str = START_PAD, end_pad: str = END_PAD) -> typing.Sequence[typing.Tuple[str, ...]]:
    tokens = ngram_preprocess(text)
    if pad > 0:
        tokens = [start_pad] * pad + tokens + [end_pad] * pad
    return list(nltk_ngrams(tokens, n))


@dataclass
class NGramTrieNode:
    children: typing.Dict[str, 'NGramTrieNode'] = field(default_factory=dict)
    count: int = 0

    def __getitem__(self, key):
        return self.children[key]

    def __setitem__(self, key, value):
        self.children[key] = value

    def __contains__(self, key):
        return key in self.children


class NGramTrieModel:
    k: typing.Optional[int]
    n: int
    should_pad: bool
    stupid_backoff_discount: float
    total_token_count: int
    tree: typing.Dict[typing.Tuple[str, ...], int]
    zero_log_prob: float

    def __init__(self, n: int, stupid_backoff_discount: float = DEFAULT_STUPID_BACKOFF_DISCOUNT,
                 zero_log_prob: float = DEFAULT_ZERO_LOG_PROB, should_pad: bool = True):
        self.n = n
        self.stupid_backoff_discount = stupid_backoff_discount
        self.zero_log_prob = zero_log_prob
        self.should_pad = should_pad

        self.k = None
        # self.root = NGramTrieNode(children={}, count=0)
        self.tree = defaultdict(int)
        self.total_token_count = 0

    def _add(self, ngram: typing.Tuple[str, ...], count: int = 1):
        # node = self.root
        # for i in range(self.n):
        #     if ngram[i] not in node:
        #         node[ngram[i]] = NGramTrieNode()

        #     node = node[ngram[i]]
        #     node.count += count

        for i in range(1, len(ngram) + 1):
            self.tree[ngram[:i]] += count

    def _add_all(self, ngrams: typing.Iterable[typing.Tuple[str, ...]], count: int = 1):
        ngrams = list(ngrams)
        self.total_token_count += (len(ngrams) - self.n + 1) * count
        for ngram in ngrams:
            self._add(ngram, count)

    def _text_to_ngrams(self, text: str, n: typing.Optional[int] = None) -> typing.Sequence[typing.Tuple[str, ...]]:
        if n is None:
            n = self.n
        return _ngrams(text, n, pad=n - 1 if self.should_pad else 0)

    def fit(self, game_texts: typing.Optional[typing.Sequence[str]] = None,
            ngram_counts: typing.Optional[typing.Dict[typing.Tuple[str, ...], int]] = None,
            n_games: typing.Optional[int] = None):

        self.root = NGramTrieNode(children={}, count=0)
        self.total_token_count = 0

        if game_texts is None and ngram_counts is None:
            raise ValueError('Must provide either game_texts or ngram_counts')

        if game_texts is not None and ngram_counts is not None:
            raise ValueError('Must provide either game_texts or ngram_counts, not both')

        if game_texts is not None:
            for text in game_texts:
                self._add_all(self._text_to_ngrams(text))

        if ngram_counts is not None:
            if n_games is None:
                raise ValueError('Must provide n_games if ngram_counts is provided')

            for ngram, count in ngram_counts.items():
                self._add(ngram, count)

            self.total_token_count = sum([v.count for v in self.root.children.values()]) + ((self.n - 1) * n_games)

    def get(self, ngram: typing.Tuple[str, ...]):
        return self.tree[ngram]
        # node = self.root
        # for i in range(min(self.n, len(ngram))):
        #     if ngram[i] not in node:
        #         return 0
        #     node = node[ngram[i]]

        # return node.count

    def _score_ngram(self, ngram: typing.Tuple[str, ...], stupid_backoff: bool = True, log: bool = False, stupid_backoff_discount: typing.Optional[float] = None):
        if stupid_backoff:
            if stupid_backoff_discount is None:
                stupid_backoff_discount = self.stupid_backoff_discount

            discount_factor = 1.0
            start_index = 0
            n = min(self.n, len(ngram))
            ret_val = 0
            ngram_count = 0

            while start_index < n:
                ngram_count = self.get(ngram[start_index:])
                if ngram_count > 0:
                    break
                start_index += 1
                discount_factor *= stupid_backoff_discount

            if start_index == n:
                ret_val = 0

            elif start_index == n - 1:
                ret_val =  discount_factor * ngram_count / self.total_token_count

            else:
                ret_val = discount_factor * ngram_count / self.get(ngram[start_index:-1])

            if log:
                return np.log(ret_val) if ret_val > 0 else self.zero_log_prob

            return ret_val

        else:
            return self.get(ngram) / self.get(ngram[:-1])

    def _transform_ngrams(self, ngrams: typing.Sequence[typing.Tuple[str, ...]],
                  stupid_backoff: bool = True, log: bool = False,
                  stupid_backoff_discount: typing.Optional[float] = None,
                  reduction: str = 'mean'):

        if len(ngrams) == 0:
            return None

        scores = [self._score_ngram(ngram, stupid_backoff, log, stupid_backoff_discount) for ngram in ngrams]
        if reduction == 'mean':
            return np.mean(scores)

        return scores

    def transform(self, game_texts: typing.Sequence[str], stupid_backoff: bool = True, log: bool = False,
                  stupid_backoff_discount: typing.Optional[float] = None):
        return np.array([self._transform_ngrams(self._text_to_ngrams(text), stupid_backoff, log, stupid_backoff_discount)
                         for text in game_texts])

    def fit_transform(self, game_texts: typing.Sequence[str]):
        self.fit(game_texts)
        return self.transform(game_texts)

    def find_ngram_counts(self, filter_padding: bool = False, min_length: int = 2, max_length: typing.Optional[int] = None):
        if max_length is None:
            max_length = self.n

        # counts = {n: dict() for n in range(min_length, max_length + 1)}
        # self._find_ngram_counts(self.root, [], counts, min_length=min_length, max_length=max_length)
        counts = {n: dict() for n in range(min_length, max_length + 1)}
        for ngram, count in self.tree.items():
            n = len(ngram)
            if (min_length <= n <= max_length) and (not filter_padding or (START_PAD not in ngram and END_PAD not in ngram)):
                counts[n][ngram] = count

        return counts

    # def _find_ngram_counts(self,
    #                     #    node: NGramTrieNode, ngram: typing.List[str],
    #                     #    ngram_counts_by_length: typing.Dict[int, typing.Dict[typing.Tuple[str, ...], int]],
    #                        filter_padding: bool = False,
    #                        min_length: int = 2,
    #                        max_length: typing.Optional[int] = None,
    #                        ):

    #     if max_length is None:
    #         max_length = self.n

        # current_length = len(ngram)
        # if min_length <= current_length <= max_length:
        #     ngram_counts_by_length[current_length][tuple(ngram)] = node.count

        # if current_length < max_length:
        #     for child, child_node in node.children.items():
        #         self._find_ngram_counts(child_node, ngram + [child], ngram_counts_by_length)

    def _get_dict_item_value(self, item: typing.Tuple[typing.Tuple[str, ...], int]):
        return item[1]

    def score(self, input_text: typing.Optional[str] = None,
              input_ngrams: typing.Optional[typing.Dict[int, typing.Sequence[typing.Tuple[str, ...]]]] = None,
              k: typing.Optional[int] = None, stupid_backoff: bool = True, log: bool = False,
              filter_padding_top_k: bool = True, top_k_min_n: typing.Optional[int] = None,
              top_k_max_n: typing.Optional[int] = None, score_all: bool = False,
              stupid_backoff_discount: typing.Optional[float] = None):

        if input_text is None and input_ngrams is None:
            raise ValueError('Must provide either text or ngrams')

        if input_text is not None and input_ngrams is not None:
            raise ValueError('Must provide either text or ngrams, not both')

        use_top_k = False
        if top_k_min_n is not None or top_k_max_n is not None:
            use_top_k = True
            if top_k_min_n is None:
                top_k_min_n = 2

            if top_k_max_n is None:
                top_k_max_n = self.n

        if input_text is not None:
            if use_top_k:
                input_ngrams = {n: list(self._text_to_ngrams(input_text)) for n in range(top_k_min_n, top_k_max_n + 1)}  # type: ignore
            else:
                input_ngrams = {self.n: list(self._text_to_ngrams(input_text))}

        input_ngrams = typing.cast(typing.Dict[int, typing.Sequence[typing.Tuple[str, ...]]], input_ngrams)

        if use_top_k and score_all:
            output = {f'n_{n}_score': self._transform_ngrams(input_ngrams[n], stupid_backoff, log, stupid_backoff_discount, reduction='mean')
                      for n in range(top_k_min_n, top_k_max_n + 1)}  # type: ignore
        else:
            output = dict(score=self._transform_ngrams(input_ngrams[self.n], stupid_backoff, log, stupid_backoff_discount, reduction='mean'))

        if k is not None:
            if k != self.k:
                self.k = k
                self.ngram_counts = self.find_ngram_counts(filter_padding=filter_padding_top_k)
                self.top_k_ngrams = {n: sorted(n_counts.items(), key=self._get_dict_item_value, reverse=True)[:k]
                                     for n, n_counts in self.ngram_counts.items()}

            if not use_top_k:
                if len(input_ngrams[self.n]) > 0:
                    text_ngram_counts = Counter(input_ngrams[self.n])
                    for i, (ngram, _) in enumerate(self.top_k_ngrams[self.n]):
                        output[i] = text_ngram_counts[ngram]  # type: ignore

            else:
                for n in range(top_k_min_n, top_k_max_n + 1):  # type: ignore
                    if len(input_ngrams[n]) > 0:
                        text_ngram_counts = Counter(input_ngrams[n])
                        for i, (ngram, _) in enumerate(self.top_k_ngrams[n]):
                            output[f'n_{n}_{i}'] = text_ngram_counts[ngram]  # type: ignore

        return output


# class BaseNGramModel:
#     def __init__(self, n: int, default_logprob: typing.Optional[float] = None):
#         self.n = n
#         self.default_logprob = default_logprob
#         self.k = None
#         self.top_k_ngrams = None

#     def _default_logprob(self):
#         return self.default_logprob

#     def _compute_ngram_counts(self, game_texts: typing.Sequence[str]):
#         return Counter(itertools.chain.from_iterable(_ngrams(text, self.n) for text in game_texts))

#     def fit(self, game_texts: typing.Optional[typing.Sequence[str]] = None,
#             ngram_counts: typing.Optional[typing.Dict[typing.Tuple[str, ...], int]] = None):

#         if game_texts is None and ngram_counts is None:
#             raise ValueError('Must provide either game_texts or ngram_counts')

#         if game_texts is not None and ngram_counts is not None:
#             raise ValueError('Must provide either game_texts or ngram_counts, not both')

#         if game_texts is not None:
#             ngram_counts = self._compute_ngram_counts(game_texts)

#         self.ngram_counts = typing.cast(typing.Dict[typing.Tuple[str, ...], int], ngram_counts)
#         self.total_ngram_counts = sum(self.ngram_counts.values())
#         if self.default_logprob is None:
#             self.default_logprob = np.log(1 / self.total_ngram_counts)
#         self.ngram_logprobs = defaultdict(self._default_logprob, {ngram: np.log(count / self.total_ngram_counts) for ngram, count in self.ngram_counts.items()})

#     def _text_to_ngrams(self, text: str) -> typing.Iterable[typing.Tuple[str, ...]]:
#         return nltk_ngrams(ngram_preprocess(text).split(), self.n)

#     def _transform_ngrams(self, ngrams: typing.Iterable[typing.Tuple[str, ...]], exp: bool = False):
#         mean_logprob = np.mean([self.ngram_logprobs[ngram] for ngram in ngrams])
#         if exp:
#             return np.exp(mean_logprob)
#         return mean_logprob

#     def transform(self, game_texts: typing.Sequence[str], exp: bool = False):
#         return np.array([self._transform_ngrams(self._text_to_ngrams(text), exp) for text in game_texts])

#     def fit_transform(self, game_texts: typing.Sequence[str]):
#         self.fit(game_texts)
#         return self.transform(game_texts)

#     def _get_dict_item_value(self, item: typing.Tuple[typing.Tuple[str, ...], int]):
#         return item[1]

#     def score(self, input_text: typing.Optional[str] = None,
#               input_ngrams: typing.Optional[typing.Iterable[typing.Tuple[str, ...]]] = None,
#               k: typing.Optional[int] = None, exp: bool = False):

#         if input_text is None and input_ngrams is None:
#             raise ValueError('Must provide either text or ngrams')

#         if input_text is not None and input_ngrams is not None:
#             raise ValueError('Must provide either text or ngrams, not both')

#         if input_text is not None:
#             input_ngrams = self._text_to_ngrams(input_text)

#         input_ngrams = list(input_ngrams)  # type: ignore
#         output = dict(score=self._transform_ngrams(input_ngrams, exp=exp))
#         if k is not None:
#             text_ngram_counts = Counter(input_ngrams)
#             if k != self.k:
#                 self.k = k
#                 self.top_k_ngrams = sorted(self.ngram_counts.items(), key=self._get_dict_item_value, reverse=True)[:k]

#             for i, (ngram, _) in enumerate(self.top_k_ngrams):  # type: ignore
#                 output[i] = text_ngram_counts[ngram]  # type: ignore

#         return output


# class TextMultiNGramModel:
#     def __init__(self, n_values: typing.Sequence[int], default_logprob: typing.Union[None, float, typing.Sequence[typing.Union[float, None]]] = None):
#         self.n_values = n_values
#         self.default_logprob = default_logprob

#         if default_logprob is None:
#             default_logprob = [None] * len(n_values)
#         elif isinstance(default_logprob, float):
#             default_logprob = [default_logprob] * len(n_values)
#         else:
#             assert len(default_logprob) == len(n_values)

#         self.models = [BaseNGramModel(n, dlp) for n, dlp in zip(n_values, default_logprob)]

#     def fit(self, game_texts: typing.Sequence[str]):
#         for model in self.models:
#             model.fit(game_texts)

#     def transform(self, game_texts: typing.Sequence[str]):
#         return np.array([model.transform(game_texts) for model in self.models])

#     def fit_transform(self, game_texts: typing.Sequence[str]):
#         self.fit(game_texts)
#         return self.transform(game_texts)

#     def score(self, text: str, k: typing.Optional[int] = None):
#         output_dict = {}
#         for model in self.models:
#             model_output = model.score(input_text=text, k=k)
#             output_dict.update({f'n_{model.n}_{key}': value for key, value in model_output.items()})

#         return output_dict


IGNORE_RULES = [
    'setup', 'setup_statement',
    'variable_list',
    'type_definition', 'color_type_definition', 'orientation_type_definition', 'side_type_definition',
    'either_types', 'either_color_types', 'either_orientation_types', 'either_side_types',
    'object_type', 'object_name', 'color_type', 'color', 'orientation_type', 'orientation', 'side_type', 'side',
    'super_predicate', 'predicate', 'function_eval', 'comparison_arg',
    'pref_forall_prefs', 'pref_def', 'pref_body', 'seq_func',
    'terminal', 'terminal_expr', 'scoring_expr', 'scoring_expr_or_number',
    'preference_eval', 'pref_object_type',
    'comparison_arg_number_value', 'time_number_value', 'score_number_value', 'scoring_number_value',
]

# avoid shuffling: then.then_funcs, while_hold.while_preds, pref_name_and_types.object_types
SHUFFLE_LIST_ELEMENTS_IGNORE_RULES = [
    'then', 'while_hold', 'pref_name_and_types',
]


class NGramASTParser(ast_parser.ASTParser):
    def __init__(self, n: int, ignore_rules: typing.Sequence[str] = IGNORE_RULES,
                 use_specific_objects: bool = False,
                 preorder_traversal: bool = True, pad: int = 0,
                 skip_game_and_domain: bool = True,
                 shuffle_list_elements_ignore_rules: typing.Sequence[str] = SHUFFLE_LIST_ELEMENTS_IGNORE_RULES,
                 random_seed: int = DEFAULT_RANDOM_SEED):
        self.n = n
        self.ignore_rules = set(ignore_rules)
        self.use_specific_objects = use_specific_objects
        self.preorder_traversal = preorder_traversal
        self.pad = pad
        self.skip_game_and_domain = skip_game_and_domain
        self.shuffle_list_elements_ignore_rules = shuffle_list_elements_ignore_rules
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

        self.ngram_counts = defaultdict(int)
        self.ngram_counts_by_section = {section: defaultdict(int) for section in ast_parser.SECTION_KEYS}
        self.current_input_ngrams = {}
        self.current_input_ngrams_by_section = {section: {} for section in ast_parser.SECTION_KEYS}
        self.preorder_ast_tokens = []
        self.preorder_ast_tokens_by_section = {section: [] for section in ast_parser.SECTION_KEYS}
        self.ast_counts_by_section = defaultdict(int)

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        # To allow unpickling of old models
        self.__dict__.update(state)
        if not hasattr(self, 'use_specific_objects'):
            self.use_specific_objects = False

        if not hasattr(self, 'shuffle_list_elements_ignore_rules'):
            self.shuffle_list_elements_ignore_rules = SHUFFLE_LIST_ELEMENTS_IGNORE_RULES

    def _add_token(self, token: str, at_start: bool = False, **kwargs):
        if at_start:
            self.preorder_ast_tokens.insert(0, token)
        else:
            self.preorder_ast_tokens.append(token)

        if ast_parser.SECTION_CONTEXT_KEY in kwargs:
            if at_start:
                self.preorder_ast_tokens_by_section[kwargs[ast_parser.SECTION_CONTEXT_KEY]].insert(0, token)
            else:
                self.preorder_ast_tokens_by_section[kwargs[ast_parser.SECTION_CONTEXT_KEY]].append(token)

    def _add_tokens(self, tokens: typing.Sequence[str], at_start: bool = False, **kwargs):
        for token in tokens:
            self._add_token(token, at_start=at_start, **kwargs)

    def parse_test_input(self, test_ast: typing.Union[tatsu.ast.AST, tuple],
                         n_values: typing.Optional[typing.Sequence[int]] = None, **kwargs) -> typing.Tuple[typing.Dict[int, typing.List[typing.Tuple[str, ...]]],
                                                                                                           typing.Dict[str, typing.Dict[int, typing.List[typing.Tuple[str, ...]]]]]:
        if n_values is not None:
            self.current_input_ngrams = {n: [] for n in n_values}
            self.current_input_ngrams_by_section = {section: {n: [] for n in n_values} for section in ast_parser.SECTION_KEYS}
            self(test_ast, update_model_counts=False, n_values=n_values, **kwargs)

        else:
            self.current_input_ngrams = {self.n: []}
            self.current_input_ngrams_by_section = {section: {self.n: []} for section in ast_parser.SECTION_KEYS}
            self(test_ast, update_model_counts=False, **kwargs)

        return self.current_input_ngrams, self.current_input_ngrams_by_section

    def __call__(self, ast, **kwargs):
        self._default_kwarg(kwargs, 'update_model_counts', False)
        self._default_kwarg(kwargs, 'skip_game_and_domain', self.skip_game_and_domain)
        self._default_kwarg(kwargs, 'shuffle_list_elements', False)
        initial_call = kwargs.get('initial_call', True)
        if initial_call:
            kwargs['initial_call'] = False
            self.preorder_ast_tokens = []
            self.preorder_ast_tokens_by_section = {section: [] for section in ast_parser.SECTION_KEYS}

        if initial_call:
            if kwargs['skip_game_and_domain']:
                ast = ast[3:]

        super().__call__(ast, **kwargs)

        if initial_call:
            if ast_parser.SECTION_CONTEXT_KEY in kwargs:
                del kwargs[ast_parser.SECTION_CONTEXT_KEY]

            if self.pad > 0:
                self._add_tokens([START_PAD] * self.pad, at_start=True)
                for section in self.preorder_ast_tokens_by_section:
                    self.preorder_ast_tokens_by_section[section] = ([START_PAD] * self.pad) + self.preorder_ast_tokens_by_section[section]

                self._add_tokens([END_PAD] * self.pad)
                for section in self.preorder_ast_tokens_by_section:
                    self.preorder_ast_tokens_by_section[section] = self.preorder_ast_tokens_by_section[section] + ([END_PAD] * self.pad)

            if 'n_values' in kwargs:
                for n in kwargs['n_values']:
                    self._count_ngrams_from_tokens(self.preorder_ast_tokens, n, update_model_counts=kwargs['update_model_counts'])
                    for section in self.preorder_ast_tokens_by_section:
                        self._count_ngrams_from_tokens(self.preorder_ast_tokens_by_section[section], n, update_model_counts=kwargs['update_model_counts'], section=section)

            else:
                self._count_ngrams_from_tokens(self.preorder_ast_tokens, self.n, update_model_counts=kwargs['update_model_counts'])
                for section in self.preorder_ast_tokens_by_section:
                    self._count_ngrams_from_tokens(self.preorder_ast_tokens_by_section[section], self.n, update_model_counts=kwargs['update_model_counts'], section=section)

    def _count_ngrams_from_tokens(self, tokens: typing.List[str], n: int, update_model_counts: bool = True, section: typing.Optional[str] = None):
        for start_index in range(len(tokens) - n + 1):
            ngram = tuple(tokens[start_index:start_index + n])
            if all(t == START_PAD or t == END_PAD for t in ngram):
                continue

            if update_model_counts:
                if section is None:
                    self.ngram_counts[ngram] += 1
                else:
                    self.ast_counts_by_section[section] += 1
                    self.ngram_counts_by_section[section][ngram] += 1
            else:
                if section is None:
                    self.current_input_ngrams[n].append(ngram)
                else:
                    self.current_input_ngrams_by_section[section][n].append(ngram)

    def _map_to_type_or_category(self, term_or_terms: typing.Union[str, typing.List[str]],
        context_variables: typing.Dict[str, typing.Union[ast_parser.VariableDefinition, typing.List[ast_parser.VariableDefinition]]]) -> typing.Union[typing.Literal['<unknown>'], typing.List[str]]:

        if self.use_specific_objects:
            types_or_categories = ast_parser.predicate_function_term_to_types(term_or_terms, context_variables)
        else:
            types_or_categories = ast_parser.predicate_function_term_to_type_categories(term_or_terms, context_variables, {})

        if types_or_categories is None or len(types_or_categories) == 0:
            return UNKNOWN_CATEGORY

        return types_or_categories

    def _combine_either_types_str(self, types: typing.List[str]) -> str:
        return f'either_types_{"_".join(sorted(types))}'

    def _combine_either_types_list(self, types: typing.List[str]) -> typing.List[str]:
        return ['either_types'] + types

    def _tokenize_ast_node(self, ast: tatsu.ast.AST, **kwargs) -> typing.Union[str, typing.List[str]]:
        rule = ast.parseinfo.rule  # type: ignore
        types_or_categories = None
        found_types_or_categories = False
        combine_types_func = None

        if rule.startswith('predicate_or_function_') and rule.endswith('term'):
            term = ast.term
            if isinstance(term, tatsu.ast.AST):
                term = term.terminal

            term = typing.cast(str, term)

            local_variables = kwargs[ast_parser.VARIABLES_CONTEXT_KEY] if ast_parser.VARIABLES_CONTEXT_KEY in kwargs else {}
            types_or_categories = self._map_to_type_or_category(term, local_variables)
            found_types_or_categories = True

            combine_types_func = self._combine_either_types_str

        if rule.endswith('variable_type_def'):
            var_type = ast.var_type.type  # type: ignore
            var_type_rule = var_type.parseinfo.rule  # type: ignore

            var_types = ast_parser._extract_variable_type_as_list(var_type)  # type: ignore
            if len(var_types) == 1:
                types_or_categories = self._map_to_type_or_category(var_types, {})
            else:
                types_or_categories = list(itertools.chain.from_iterable(self._map_to_type_or_category(var_type, {}) for var_type in var_types))

            if var_type_rule.startswith('either'):
                combine_types_func = self._combine_either_types_list

            found_types_or_categories = True

        if found_types_or_categories:
            if types_or_categories is None or len(types_or_categories) == 0 or types_or_categories == UNKNOWN_CATEGORY:
                return UNKNOWN_CATEGORY

            types_or_categories = typing.cast(typing.List[str], types_or_categories)

            if len(types_or_categories) == 1:
                return types_or_categories[0]
            else:
                if combine_types_func is not None:
                    return combine_types_func(types_or_categories)
                else:
                    raise ValueError(f'Found multiple types/categories for {ast} with no combine_types_func: {types_or_categories}')

        if rule == 'pref_name_and_types':
            output = ['pref_name']
            if ast.object_types is not None and len(ast.object_types) > 0:
                object_types = ast.object_types
                if not isinstance(object_types, (list, tuple)):
                    object_types = [object_types]

                object_types = [t.type_name.terminal if isinstance(t, tatsu.ast.AST) else str(t) for t in object_types]
                for obj_type in object_types:
                    object_category = self._map_to_type_or_category(obj_type, {})  # type: ignore
                    if object_category is None or len(object_category) == 0:
                        raise ValueError(f'Could not find category for object type {obj_type}: {object_category}')

                    obj = object_category if isinstance(object_category, str) else f'object_type_{object_category[0]}'
                    output.append(obj)  # type: ignore

            return output

        if rule in ('terminal_comp', 'scoring_comp', 'scoring_multi_expr'):
            if rule == "terminal_comp":
                ast = typing.cast(tatsu.ast.AST, ast.comp)

            return ast.op  # type: ignore

        if rule == 'two_arg_comparison':
            return ast.comp_op  # type: ignore

        if rule in ('scoring_equals_comp', 'multiple_args_equal_comparison'):
            return '='

        if rule == 'scoring_neg_expr':
            return '-'

        if rule in ('comparison_arg_number_value', 'time_number_value', 'score_number_value', 'scoring_number_value'):
            return 'number'

        return rule

    def _handle_tuple(self, ast: tuple, **kwargs):
        if isinstance(ast[0], str) and ast[0].startswith('(:'):
            section = ast[0]
            kwargs[ast_parser.SECTION_CONTEXT_KEY] = section
            self._add_token(section[2:], **kwargs)

        super()._handle_tuple(ast, **kwargs)

    def _handle_str(self, ast: str, **kwargs):
        token = None
        if NUMBER_AND_DECIMAL_PATTERN.match(ast):
            token = 'number'
        elif ast.startswith('(total-'):
            token = ast.replace('(', '').replace(')', '')

        if token is not None:
            self._add_token(token, **kwargs)

    def _handle_int(self, ast: int, **kwargs):
        self._add_token('number', **kwargs)

    def _handle_ast(self, ast: tatsu.ast.AST, **kwargs):
        rule = ast.parseinfo.rule  # type: ignore
        kwargs = ast_parser.update_context_variables(ast, kwargs)

        if rule not in self.ignore_rules:
            token = self._tokenize_ast_node(ast, **kwargs)
            if isinstance(token, list):
                self._add_tokens(token, **kwargs)
            else:
                self._add_token(token, **kwargs)

        if rule in self.shuffle_list_elements_ignore_rules:
            kwargs['skip_shuffle'] = True

        for child_key in ast:
            if child_key != 'parseinfo':
                self(ast[child_key], **kwargs)

    def _handle_list(self, ast: list, **kwargs):
        if kwargs['shuffle_list_elements'] and not kwargs.pop('skip_shuffle', False):
            self.rng.shuffle(ast)

        return super()._handle_list(ast, **kwargs)


DEFAULT_N_BY_SECTION = {
    ast_parser.SETUP: 5,
    ast_parser.PREFERENCES: 7,
    ast_parser.TERMINAL: 5,
    ast_parser.SCORING: 5,
}


class ASTNGramTrieModel:
    def __init__(self, n: int, ignore_rules: typing.Sequence[str] = IGNORE_RULES,
                 use_specific_objects: bool = False,
                 stupid_backoff_discount: float = DEFAULT_STUPID_BACKOFF_DISCOUNT,
                 zero_log_prob: float = DEFAULT_ZERO_LOG_PROB,
                 preorder_traversal: bool = True, pad: int = 0,
                 fit_shuffle_list_elements: bool = False, random_seed: int = DEFAULT_RANDOM_SEED,
                 n_by_section: typing.Dict[str, int] = DEFAULT_N_BY_SECTION,
                 sections: typing.Sequence[str] = ast_parser.SECTION_KEYS):

        self.n = n
        self.ignore_rules = ignore_rules
        self.use_specific_objects = use_specific_objects
        self.fit_shuffle_list_elements = fit_shuffle_list_elements
        self.random_seed = random_seed
        self.sections = sections
        for section in sections:
            if section not in n_by_section:
                n_by_section[section] = n
        self.n_by_sections = n_by_section

        self.ngram_ast_parser = NGramASTParser(n, ignore_rules=ignore_rules, use_specific_objects=use_specific_objects,
                                               preorder_traversal=preorder_traversal, pad=pad, random_seed=random_seed)
        self.model = NGramTrieModel(n, stupid_backoff_discount=stupid_backoff_discount, zero_log_prob=zero_log_prob, should_pad=False)
        self.model_by_section = {section: NGramTrieModel(self.n_by_sections[section], stupid_backoff_discount=stupid_backoff_discount, zero_log_prob=zero_log_prob, should_pad=False) for section in sections}

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        # To allow unpickling of old models
        self.__dict__.update(state)
        if not hasattr(self, 'use_specific_objects'):
            self.use_specific_objects = False

    def fit(self, asts: typing.Sequence[typing.Union[tuple,tatsu.ast.AST]]):
        for ast in asts:
            self.ngram_ast_parser(ast, update_model_counts=True, shuffle_list_elements=self.fit_shuffle_list_elements)

        self.model.fit(ngram_counts=self.ngram_ast_parser.ngram_counts, n_games=len(asts))
        for section, model in self.model_by_section.items():
            model.fit(ngram_counts=self.ngram_ast_parser.ngram_counts_by_section[section], n_games=self.ngram_ast_parser.ast_counts_by_section[section])

    def score(self, ast: typing.Union[tuple,tatsu.ast.AST], k: typing.Optional[int] = None,
              stupid_backoff: bool = True, log: bool = False,
              filter_padding_top_k: bool = True, top_k_min_n: typing.Optional[int] = None,
              top_k_max_n: typing.Optional[int] = None, k_for_sections: typing.Optional[int] = None,
              score_all: bool = False, tokenize_entire_ast: bool = False,
              ngram_ast_parser_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
              debug: bool = False):

        if ngram_ast_parser_kwargs is None:
            ngram_ast_parser_kwargs = {}

        if k_for_sections is None:
            k_for_sections = k

        n_values = None
        if top_k_min_n is not None:
            if top_k_max_n is None:
                top_k_max_n = self.n

            n_values = list(range(top_k_min_n, top_k_max_n + 1))

        if 'skip_game_and_domain' not in ngram_ast_parser_kwargs:
            ngram_ast_parser_kwargs['skip_game_and_domain'] = not tokenize_entire_ast

        current_input_ngrams, current_input_ngrams_by_section = self.ngram_ast_parser.parse_test_input(
            ast, n_values=n_values, **ngram_ast_parser_kwargs)

        if debug: print(current_input_ngrams[2])

        outputs = {'full': self.model.score(input_ngrams=current_input_ngrams, k=k,  # type: ignore
                                stupid_backoff=stupid_backoff, log=log,
                                filter_padding_top_k=filter_padding_top_k,
                                top_k_min_n=top_k_min_n, top_k_max_n=top_k_max_n,
                                score_all=score_all)}

        for section in self.sections:
            outputs[section.replace('(:', '')] = self.model_by_section[section].score(input_ngrams=current_input_ngrams_by_section[section], k=k_for_sections,  # type: ignore
                                                                    stupid_backoff=stupid_backoff, log=log,
                                                                    filter_padding_top_k=filter_padding_top_k,
                                                                    top_k_min_n=top_k_min_n, top_k_max_n=None if top_k_max_n is None else min(top_k_max_n, self.n_by_sections[section]),
                                                                    score_all=score_all)


        return {f'{section}_{key}': value for section, output in outputs.items() for key, value in output.items()}


def main(args: argparse.Namespace):
    if args.from_asts:
        model = ASTNGramTrieModel(n=args.n, use_specific_objects=args.use_specific_objects,
                                  stupid_backoff_discount=args.stupid_backoff_discount,
                                  zero_log_prob=args.zero_log_prob, pad=0 if args.no_pad else args.padding,
                                  fit_shuffle_list_elements=args.fit_shuffle_list_elements,)
    else:
        model = NGramTrieModel(n=args.n, stupid_backoff_discount=args.stupid_backoff_discount, zero_log_prob=args.zero_log_prob)

    grammar = open(args.grammar_file).read()
    grammar_parser = tatsu.compile(grammar)

    game_inputs = []
    for test_file in args.test_files:
        if args.from_asts:
            game_inputs.extend(cached_load_and_parse_games_from_file(test_file, grammar_parser, True))  # type: ignore
        else:
            game_inputs.extend(ast_printer.ast_to_string(ast, '\n') for ast in cached_load_and_parse_games_from_file(test_file, grammar_parser, False))  # type: ignore

    model.fit(game_inputs)

    # [print(model.score(game)) for game in game_inputs]

    with open(args.output_path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    args = parser.parse_args()

    # if not args.n:
    #     args.n = [DEFAULT_N]

    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    if args.output_path is None:
        model_type = 'ast' if args.from_asts else 'text'
        if args.use_specific_objects:
            model_type += '_specific_objects'
        args.output_path = DEFAULT_OUTPUT_PATH_PATTERN.format(model_type=model_type,
            n=args.n, today=datetime.now().strftime('%Y_%m_%d'))

    if args.padding is None:
        args.padding = args.n - 1

    main(args)
