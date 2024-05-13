import argparse
from collections import Counter
import gzip
import json
import logging
import os
import pickle
from pprint import pformat
import sys
import typing

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, ParameterGrid
import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import ast_printer  # for logging
import ast_parser  # for logging
from src import fitness_energy_utils as utils
from src import latest_model_paths
from src.fitness_ngram_models import *
from src.ast_utils import cached_load_and_parse_games_from_file, CACHE_FOLDER

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


parser = argparse.ArgumentParser()
DEFAULT_GRAMMAR_FILE = './dsl/dsl.ebnf'
parser.add_argument('-g', '--grammar-file', default=DEFAULT_GRAMMAR_FILE)
DEFAULT_REAL_GAMES_FILE = './dsl/interactive-beta.pddl'
parser.add_argument('-real-games-file',type=str, default=DEFAULT_REAL_GAMES_FILE)
DEFAULT_REGROWTHS_FILE = './dsl/ast-real-regrowth-samples-32.pddl.gz'
parser.add_argument('-regrowths-file',type=str, default=DEFAULT_REGROWTHS_FILE)
parser.add_argument('--min-n', type=int, default=2)
parser.add_argument('--max-n', type=int, default=7)
parser.add_argument('--evaluate-sections', action='store_true')

parser.add_argument('--pad', action='store_true')
parser.add_argument('--use-specific-objects', action='store_true')
parser.add_argument('--cv-settings-json', type=str, default=os.path.join(os.path.dirname(__file__), 'ngram_cv_settings.json'))
DEFAULT_N_FOLDS = 5
parser.add_argument('--cv-n-folds', type=int, default=DEFAULT_N_FOLDS)
parser.add_argument('--refit-metric', type=str, default='real')
DEFAULT_RANDOM_SEED = 33
parser.add_argument('--random-seed', type=int, default=DEFAULT_RANDOM_SEED)

parser.add_argument('--tokenize-games', action='store_true')
DEFAULT_TOKENIZED_GAMES_FILE_NAME = 'ngram_tokenized_games.pkl.gz'
parser.add_argument('--tokenized-games-file-name', type=str, default=DEFAULT_TOKENIZED_GAMES_FILE_NAME)
parser.add_argument('--regrowths-per-real-game', type=int, default=32)


FULL_GAME_SECTION_KEY = 'full'

TokenizedGame = typing.List[tuple]


class NgramSklearnWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n: int, stupid_backoff_discount: float, zero_log_prob: float, log: bool = True, **params):
        self.n = n
        self.stupid_backoff_discount = stupid_backoff_discount
        self.zero_log_prob = zero_log_prob
        self.log = log

        self.model = None

    # def get_params(self, deep: bool = True) -> typing.Dict[str, typing.Any]:
    #     return self.params

    # def set_params(self, **params: typing.Dict[str, typing.Any]) -> 'NgramSklearnWrapper':
    #     if 'params' in params:
    #         self.params.update(params['params'])
    #         del params['params']

    #     self.params.update(params)

    #     if 'n' in self.params:
    #         self.n = self.params['n']

    #     return self

    def fit(self, X: typing.List[typing.List[TokenizedGame]], y=None) -> 'NgramSklearnWrapper':
        real_games, regrown_games = zip(*[(x[0], x[1:]) for x in X])

        self.model = NGramTrieModel(self.n, stupid_backoff_discount=self.stupid_backoff_discount, zero_log_prob=self.zero_log_prob)
        ngram_counts = Counter()
        for game in real_games:
            ngram_counts.update(game)

        self.model.fit(ngram_counts=ngram_counts, n_games=len(real_games))
        return self

    def _score(self, game_ngrams: TokenizedGame) -> float:
        return self.model.score(input_ngrams={self.n: game_ngrams}, top_k_min_n=self.n, top_k_max_n=self.n, log=self.log, score_all=True)[f'n_{self.n}_score']  # type: ignore

    def transform(self, X: typing.List[typing.List[TokenizedGame]], y=None):
        if self.model is None:
            raise ValueError('Must call fit() before calling transform()')

        real_games, regrown_games = zip(*[(x[0], x[1:]) for x in X])
        real_game_scores = [self._score(game_ngrams) for game_ngrams in real_games]  # type: ignore
        regrown_game_scores = [[self._score(game_ngrams) for game_ngrams in regrown_games] for regrown_games in regrown_games]  # type: ignore
        return real_game_scores, regrown_game_scores



def score_real_games(model: NgramSklearnWrapper, X: typing.List[typing.List[TokenizedGame]], y=None) -> float:
    real_game_scores, _ = model.transform(X)
    real_game_scores = np.array(real_game_scores, dtype=float)
    return -real_game_scores.mean()


def score_regrown_games(model: NgramSklearnWrapper, X: typing.List[typing.List[TokenizedGame]], y=None) -> float:
    _, regrown_game_scores = model.transform(X)
    regrown_game_scores = np.array(regrown_game_scores, dtype=float)
    return -regrown_game_scores.mean()


def score_regrowth_differences(model: NgramSklearnWrapper, X: typing.List[typing.List[TokenizedGame]], y=None) -> float:
    real_game_scores, regrown_game_scores = model.transform(X)
    real_game_scores = np.array(real_game_scores, dtype=float)
    regrown_game_scores = np.array(regrown_game_scores, dtype=float)
    # # return -(real_game_scores.mean() - regrown_game_scores.mean())
    return -np.nanmean(real_game_scores - np.nanmean(regrown_game_scores, axis=1))


SCORE_KEYS = ['real', 'regrown', 'regrowth_differences']


def combined_score(model: NgramSklearnWrapper, X: typing.List[typing.List[TokenizedGame]], y=None) -> typing.Dict[str, float]:
    return {
        'real': score_real_games(model, X, y),
        'regrown': score_regrown_games(model, X, y),
        'regrowth_differences': score_regrowth_differences(model, X, y),
    }


def _create_tokenized_games_dict(n_values: typing.List[int]) -> typing.Dict[str, typing.Dict[int, typing.Dict[str, typing.List[TokenizedGame]]]]:
    return {
        section: {n: {'real': [], 'regrown': []} for n in n_values}
        for section in [FULL_GAME_SECTION_KEY] + list(ast_parser.SECTION_KEYS)
    }


def single_key_evaluation(
        section_key: str, args: argparse.Namespace,
        tokenized_games: typing.Dict[str, typing.Dict[int, typing.Dict[str, typing.List[TokenizedGame]]]],
        cv_settings: typing.Dict[str, typing.Any], score_keys: typing.List[str] = SCORE_KEYS):
    cv_dfs = []
    for n in range(args.min_n, args.max_n + 1):
        setting_tokenized_games = tokenized_games[section_key][n]
        real_games, regrown_games = setting_tokenized_games['real'], setting_tokenized_games['regrown']

        real_games_tokenized_lengts = [len(game) for game in real_games]
        games_with_section_indices = list(np.argwhere(real_games_tokenized_lengts).squeeze())

        for i in games_with_section_indices:
            regrown_games[i].insert(0, real_games[i])  # type: ignore
        all_games = [regrown_games[i] for i in games_with_section_indices]

        # logger.debug(f'CV settings:\n{pformat(cv_settings)}')
        param_grid = cv_settings['param_grid']
        param_grid['n'] = [n]

        model = NgramSklearnWrapper(None, None, None)  # type: ignore

        cv_kwargs = dict(n_jobs=-1, verbose=0, error_score='raise', refit=args.refit_metric)

        cv = GridSearchCV(model, param_grid, scoring=combined_score,
            cv=KFold(args.cv_n_folds, shuffle=True, random_state=args.random_seed),
            **cv_kwargs)  # type: ignore

        cv = cv.fit(all_games)

        cv_series_or_dfs = [
            pd.Series(cv.cv_results_[name], name=name)
            for key in score_keys
            for name in (f'mean_test_{key}', f'rank_test_{key}')
        ]
        cv_series_or_dfs.insert(0, pd.DataFrame(cv.cv_results_["params"]))  # type: ignore
        cv_df = pd.concat(cv_series_or_dfs, axis=1)
        cv_dfs.append(cv_df)

    cv_df = pd.concat(cv_dfs, axis=0)
    cv_df = cv_df.assign(**{f'rank_test_{key}': cv_df[f'mean_test_{key}'].rank() for key in score_keys})

    # cv_df['section'] = section_key
    return cv_df


def main(args: argparse.Namespace):
    if args.tokenize_games:
        args.pad = args.max_n - 1 if args.pad else 0
        ngram_ast_parser = NGramASTParser(args.max_n, ignore_rules=IGNORE_RULES, use_specific_objects=args.use_specific_objects, pad=args.pad)
        n_values = list(range(args.min_n, args.max_n + 1))

        grammar = open(args.grammar_file).read()
        grammar_parser = tatsu.compile(grammar)

        tokenized_games = _create_tokenized_games_dict(n_values)

        for real_game_ast in tqdm.tqdm(cached_load_and_parse_games_from_file(args.real_games_file, grammar_parser, False, log_every_change=False), desc='Real Games'):  # type: ignore
            full_game_tokens, tokens_by_section = ngram_ast_parser.parse_test_input(real_game_ast, n_values=n_values)  # type: ignore
            for n in range(args.min_n, args.max_n + 1):
                tokenized_games[FULL_GAME_SECTION_KEY][n]['real'].append(full_game_tokens[n])
                for section in ast_parser.SECTION_KEYS:
                    tokenized_games[section][n]['real'].append(tokens_by_section[section][n])

        current_real_game_tokenized_games = _create_tokenized_games_dict(n_values)
        current_real_game_counter = 0

        pbar = tqdm.tqdm(total=len(tokenized_games[FULL_GAME_SECTION_KEY][args.min_n]['real']), desc="Regrown Games") # type: ignore

        for regrown_game_ast in cached_load_and_parse_games_from_file(args.regrowths_file, grammar_parser, False, log_every_change=False):  # type: ignore
            full_game_tokens, tokens_by_section = ngram_ast_parser.parse_test_input(regrown_game_ast, n_values=n_values)  # type: ignore
            for n in range(args.min_n, args.max_n + 1):
                current_real_game_tokenized_games[FULL_GAME_SECTION_KEY][n]['real'].append(full_game_tokens[n])
                for section in ast_parser.SECTION_KEYS:
                    current_real_game_tokenized_games[section][n]['real'].append(tokens_by_section[section][n])

            current_real_game_counter += 1

            if current_real_game_counter >= args.regrowths_per_real_game:
                for n in range(args.min_n, args.max_n + 1):
                    tokenized_games[FULL_GAME_SECTION_KEY][n]['regrown'].append(current_real_game_tokenized_games[FULL_GAME_SECTION_KEY][n]['real'])  # type: ignore
                    for section in ast_parser.SECTION_KEYS:
                        tokenized_games[section][n]['regrown'].append(current_real_game_tokenized_games[section][n]['real'])  # type: ignore

                current_real_game_tokenized_games = _create_tokenized_games_dict(n_values)
                current_real_game_counter = 0
                pbar.update(1)


        with gzip.open(os.path.join(CACHE_FOLDER, args.tokenized_games_file_name), 'wb') as f:
            pickle.dump(tokenized_games, f)

    else:
        with gzip.open(os.path.join(CACHE_FOLDER, args.tokenized_games_file_name), 'rb') as f:
            tokenized_games = pickle.load(f)

        with open(args.cv_settings_json, 'r') as f:
            cv_settings = json.load(f)

        section_keys = [FULL_GAME_SECTION_KEY]
        if args.evaluate_sections:
            section_keys += list(ast_parser.SECTION_KEYS)

        for section_key in section_keys:
            cv_df = single_key_evaluation(section_key, args, tokenized_games, cv_settings)
            print(f'Best CV Results for section {section_key}, sorted by {args.refit_metric}:')
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_colwidth', 12)
            print(cv_df.sort_values(by=f'mean_test_{args.refit_metric}').head(10))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
