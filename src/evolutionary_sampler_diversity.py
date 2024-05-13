from functools import lru_cache
import itertools
import typing
from queue import PriorityQueue
from typing import Any

from Levenshtein import distance as edit_distance
import numpy as np
import torch

import ast_parser
import ast_printer


MAX_CACHE_SIZE = 1000000


EDIT_DISTANCE = 'edit_distance'
BY_SECTION_EDIT_DISTANCE = 'by_section_edit_distance'
BY_SECTION_EDIT_DISTANCE_MEAN = 'by_section_edit_distance_mean'
BY_SECTION_EDIT_DISTANCE_MAX = 'by_section_edit_distance_max'
BY_SECTION_EDIT_DISTANCE_MIN = 'by_section_edit_distance_min'
TENSOR_FEATURES_DISTANCE = 'tensor_features_distance'
DIVERSITY_SCORERS = (EDIT_DISTANCE, BY_SECTION_EDIT_DISTANCE, BY_SECTION_EDIT_DISTANCE_MEAN, BY_SECTION_EDIT_DISTANCE_MAX, BY_SECTION_EDIT_DISTANCE_MIN, TENSOR_FEATURES_DISTANCE)


class DiversityScorer:
    cache: typing.Dict[str, typing.Any]
    k: int
    population: typing.Dict[str, typing.Any]
    # population_pairwise_scores: typing.Dict[typing.Tuple[str, str], float]

    def __init__(self, k: int = 1):
        self.k = k

        self.cache = {}
        self.population = {}

        # self.population_pairwise_scores = {}

    def _game_to_key(self, game) -> str:
        return game[1].game_name

    def _game_pair_to_key(self, first_game, second_game) -> typing.Tuple[str, str]:
        first_key, second_key = self._game_to_key(first_game), self._game_to_key(second_game)
        return (first_key, second_key) if first_key < second_key else (second_key, first_key)

    def _key_to_game(self, game_key: str) -> typing.Any:
        if game_key in self.population:
            return self.population[game_key]
        elif game_key in self.cache:
            return self.cache[game_key]
        else:
            raise ValueError(f'Game {game_key} not found in population or cache')

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def featurize(self, game_key: str) -> typing.Any:
        return self._featurize(self._key_to_game(game_key))

    def _featurize(self, game) -> typing.Any:
        raise NotImplemented

    def score(self, first_game_key: str, second_game_key: str) -> typing.Any:
        keys = (first_game_key, second_game_key) if first_game_key < second_game_key else (second_game_key, first_game_key)
        return self._cached_score(*keys)

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _cached_score(self, first_game_key: str, second_game_key: str):
        return self._score(self.featurize(first_game_key), self.featurize(second_game_key))

    def _score(self, first_game, second_game) -> float:
        raise NotImplemented

    def set_population(self, population: typing.List[typing.Any]):
        self.population = {self._game_to_key(game): game for game in population}

    def find_most_similar_scores(self, game) -> np.ndarray:
        """
        Find the k most similar games to the given game.
        """
        game_key = self._game_to_key(game)
        if game_key not in self.population:
            self.cache[game_key] = game

        return self.find_most_similar_scores_by_key(game_key)

    def find_most_similar_scores_by_key(self, game_key: str) -> np.ndarray:
        # scores = PriorityQueue()
        # for other_game_key in self.population:
        #     if other_game_key == game_key:
        #         continue
        #     score = self.score(game_key, other_game_key)
        #     scores.put((score, other_game_key))
        # return [scores.get()[0] for _ in range(self.k)]

        scores = np.array([self.score(game_key, other_game_key) for other_game_key in self.population if other_game_key != game_key])
        return scores[np.argsort(scores)[:self.k]]

    def score_entire_population(self):
        return [self.score(first_key, second_key) for first_key, second_key in itertools.combinations(self.population, 2)]

    def population_score_distribution(self) -> np.ndarray:
        return np.array([self(game_key=game_key) for game_key in self.population])

    def __call__(self, game: typing.Optional[typing.Any] = None, game_key: typing.Optional[str] = None) -> float:
        if game_key is not None:
            return self.find_most_similar_scores_by_key(game_key).mean()

        return np.mean(self.find_most_similar_scores(game)).mean()


EDIT_DISTANCE_WEIGHTS = (1, 1, 1)


class EditDistanceDiversityScorer(DiversityScorer):
    weights: typing.Tuple[float, float, float]
    def __init__(self, k: int = 1, weights: typing.Tuple[float, float, float] = EDIT_DISTANCE_WEIGHTS, **kwargs):
        super().__init__(k=k)
        self.weights = weights

    def _featurize(self, game):
        game_string = ast_printer.ast_to_string(game)
        return game_string[game_string.find(')', game_string.find('(:domain')) + 1:]

    def _score(self, first_game, second_game):
        return edit_distance(first_game, second_game, weights=self.weights)  # type: ignore


class BySectionEditDistanceDiversityScorer(EditDistanceDiversityScorer):
    agg_func: typing.Callable[[typing.Sequence[float]], float]

    def __init__(self, agg_func: typing.Callable[[typing.Sequence[float]], float], k: int = 1, weights: typing.Tuple[float, float, float] = EDIT_DISTANCE_WEIGHTS, **kwargs):
        super().__init__(k=k, weights=weights)
        self.agg_func = agg_func

    def _featurize(self, game):
        return {
            section[0]: ast_printer.ast_section_to_string(section[1], section[0])
            for section in game[3:-1]
        }

    def _score(self, first_game, second_game):
        return self.agg_func([EditDistanceDiversityScorer._score(self, first_game[key], second_game[key]) for key in first_game if key in second_game])  # type: ignore


class TensorFeaturesDistanceDiversityScorer(DiversityScorer):
    feature_names: typing.List[str]
    featurizer: typing.Callable[..., typing.Dict[str, typing.Any]]
    ord: float

    def __init__(self, featurizer: typing.Callable[..., typing.Dict[str, typing.Any]], feature_names: typing.List[str], k: int = 1, ord: float = 2, **kwargs):
        super().__init__(k=k)
        self.featurizer = featurizer
        self.feature_names = feature_names
        self.ord = ord

    def _featurize(self, game):
        features = typing.cast(dict, self.featurizer.parse(game, return_row=True))  # type: ignore
        return torch.tensor([features[name] for name in self.feature_names], dtype=torch.float32)  # type: ignore

    def _score(self, first_game, second_game):
        return torch.linalg.vector_norm(first_game - second_game, ord=self.ord).item()


def create_diversity_scorer(scorer_type: str, **kwargs) -> DiversityScorer:
    if scorer_type == EDIT_DISTANCE:
        return EditDistanceDiversityScorer(**kwargs)
    elif scorer_type == BY_SECTION_EDIT_DISTANCE or scorer_type == BY_SECTION_EDIT_DISTANCE_MEAN:
        return BySectionEditDistanceDiversityScorer(agg_func=np.mean, **kwargs)
    elif scorer_type == BY_SECTION_EDIT_DISTANCE_MAX:
        return BySectionEditDistanceDiversityScorer(agg_func=max, **kwargs)
    elif scorer_type == BY_SECTION_EDIT_DISTANCE_MIN:
        return BySectionEditDistanceDiversityScorer(agg_func=min, **kwargs)
    elif scorer_type == TENSOR_FEATURES_DISTANCE:
        return TensorFeaturesDistanceDiversityScorer(**kwargs)
    else:
        raise ValueError(f'Unknown scorer type: {scorer_type}')
