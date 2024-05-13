import abc
import argparse
from collections import Counter, defaultdict
import copy
import enum
import itertools
from Levenshtein import distance as edit_distance
import logging
import numpy as np
import typing
import re

import tatsu
import tatsu.ast
import tatsu.grammars
from sklearn.decomposition import PCA

import ast_parser
import ast_printer
from ast_utils import cached_load_and_parse_games_from_file, deepcopy_ast, ASTCopyType
from fitness_features import ASTFitnessFeaturizer, FitnessTerm, Number, SetupObjectsUsed, ContextDict, SETUP_OBJECTS_SKIP_OBJECTS, PREDICATE_AND_FUNCTION_RULES, DEPTH_CONTEXT_KEY, SectionExistsFitnessTerm, AtEndFound, NumPreferencesDefined
from ast_counter_sampler import SamplingException
import room_and_object_types

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _bin_number(number: int, bins: np.ndarray) -> int:
    return int(np.digitize(number, bins, right=True))


NODE_COUNT_BINS = [55, 85, 103, 117, 129.5, 160, 190, 220, 300]
NODE_COUNT_BINS_8 = [61, 93, 113, 129, 168, 205, 263]


class NodeCount(FitnessTerm):
    bins: np.ndarray
    count: int = 0
    def __init__(self, bins: typing.List[int] = NODE_COUNT_BINS):
        super().__init__(re.compile('.*'), 'node_count')
        if bins is None:
            self.bins = None  # type: ignore
        else:
            self.bins = np.array(bins)

    def game_start(self) -> None:
        self.count = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        self.count += 1

    def game_end(self) -> int:
        if self.bins is not None:
            return _bin_number(self.count, self.bins)
        else:
            return self.count


UNIQUE_OBJECT_REFERENCES_BINS = [2, 3, 4, 5, 6, 7, 9, 11, 13]


class UniqueObjectsReferenced(SetupObjectsUsed):
    bins: np.ndarray
    def __init__(self, bins: typing.List[int] = UNIQUE_OBJECT_REFERENCES_BINS, skip_objects: typing.Set[str] = SETUP_OBJECTS_SKIP_OBJECTS):
        super().__init__(skip_objects=skip_objects, header='unique_objects_referenced')
        self.bins = np.array(bins)

    def game_end(self) -> int:
        return _bin_number(len(self.setup_objects.union(self.used_objects)), self.bins)


UNIQUE_PREDICATE_REFERENCES_BINS = [2, 3, 4, 5, 6, 7, 8, 9]


class UniquePredicatesReferenced(FitnessTerm):
    bins: np.ndarray
    predicates_referenced: typing.Set[str] = set()

    def __init__(self, bins: typing.List[int] = UNIQUE_PREDICATE_REFERENCES_BINS):
        super().__init__(PREDICATE_AND_FUNCTION_RULES, 'unique_predicates_referenced')
        self.bins = np.array(bins)

    def game_start(self) -> None:
        self.predicates_referenced = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            if rule == 'predicate':
                pred = ast.pred.parseinfo.rule.replace('predicate_', '')  # type: ignore

            else:
                pred = ast.func.parseinfo.rule.replace('function_', '')  # type: ignore

            self.predicates_referenced.add(pred)

    def game_end(self) -> int:
        return _bin_number(len(self.predicates_referenced), self.bins)


class MeanNodeDepth(FitnessTerm):
    node_count: int = 0
    total_depth: int = 0

    def __init__(self):
        super().__init__(re.compile('.*'), 'mean_node_depth')
        self.node_count = 0

    def game_start(self) -> None:
        self.node_count = 0
        self.total_depth = 0

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if DEPTH_CONTEXT_KEY in context:
            self.total_depth += context[DEPTH_CONTEXT_KEY]  # type: ignore
            self.node_count += 1

    def game_end(self) -> float:
        return self.total_depth / self.node_count if self.node_count > 0 else 0


SPECIFIC_PREDICATES =  ['adjacent', 'agent_holds', 'between', 'in', 'in_motion', 'on', 'touch']


def _list_to_key(pred_list: typing.List[str]) -> str:
    return '|'.join(pred_list)


class PredicateUsed(FitnessTerm):
    predicates_used: typing.Set[str]

    def __init__(self, predicates: typing.Union[typing.List[str], typing.List[typing.List[str]], typing.List[typing.Union[str, typing.List[str]]]] = SPECIFIC_PREDICATES):
        super().__init__(PREDICATE_AND_FUNCTION_RULES, 'predicate_used')
        self.predicates = predicates

    def game_start(self) -> None:
        self.predicates_used = set()

    def update(self, ast: typing.Union[typing.Sequence, tatsu.ast.AST], rule: str, context: ContextDict):
        if isinstance(ast, tatsu.ast.AST):
            if rule == 'predicate':
                pred = ast.pred.parseinfo.rule.replace('predicate_', '')  # type: ignore

            else:
                pred = ast.func.parseinfo.rule.replace('function_', '')  # type: ignore

            self.predicates_used.add(pred)

    def game_end(self):
        output = {}
        for pred_or_list in self.predicates:
            if isinstance(pred_or_list, str):
                pred = pred_or_list
                output[pred] = int(pred in self.predicates_used)
            else:
                pred_list = pred_or_list
                output[_list_to_key(pred_list)] = int(any(pred in self.predicates_used for pred in pred_list))

        return output

    def _get_all_inner_keys(self):
        return [pred_or_list if isinstance(pred_or_list, str) else _list_to_key(pred_or_list)
                for pred_or_list in self.predicates]


SPECIFIC_CATEGORIES = [room_and_object_types.BALLS, room_and_object_types.BLOCKS,
                       room_and_object_types.FURNITURE, room_and_object_types.LARGE_OBJECTS,
                       room_and_object_types.RAMPS, room_and_object_types.RECEPTACLES,
                       room_and_object_types.SMALL_OBJECTS,
                       ]

class ObjectCategoryUsed(SetupObjectsUsed):
    def __init__(self, categories: typing.Union[typing.List[str], typing.List[typing.List[str]], typing.List[typing.Union[str, typing.List[str]]]] = SPECIFIC_CATEGORIES,
                 skip_objects: typing.Set[str] = SETUP_OBJECTS_SKIP_OBJECTS):
        super().__init__(skip_objects=skip_objects, header='object_category_used', parse_variables=True)
        self.categories = categories

    def game_end(self):
        categories_used = set()
        for object_set in [self.setup_objects, self.used_objects]:
            for obj in object_set:
                if obj in room_and_object_types.TYPES_TO_CATEGORIES:
                    categories_used.add(room_and_object_types.TYPES_TO_CATEGORIES[obj])

        output = {}
        for cat_or_list in self.categories:
            if isinstance(cat_or_list, str):
                cat = cat_or_list
                output[cat] = int(cat in categories_used)
            else:
                cat_list = cat_or_list
                output[_list_to_key(cat_list)] = int(any(pred in categories_used for pred in cat_list))

        return output

    def _get_all_inner_keys(self):
        return [cat_or_list if isinstance(cat_or_list, str) else _list_to_key(cat_or_list)
                for cat_or_list in self.categories]


MAX_NUM_PREFERENCES_COUNT = 4


class NumPreferencesDefinedAsInteger(NumPreferencesDefined):
    def __init__(self, max_count: int = MAX_NUM_PREFERENCES_COUNT, min_count: int = 1):
        super().__init__(max_count=max_count, min_count=min_count)

    def game_end(self) -> Number:
        return self._get_clipped_preference_count()

    def _get_all_inner_keys(self):
        return None

    def get_n_values(self):
        return self.max_count - self.min_count + 1


class FullFeaturesFitnessTerm(FitnessTerm):
    @abc.abstractmethod
    def parse_full_features(self, features: typing.Dict[str, float]) -> None:
        pass


EXPECTED_VALUE_1_FEATURES =  (
    'variables_defined_all', 'variables_defined_prop',
    'variables_used_all', 'variables_used_prop',
    'preferences_used_all', 'preferences_used_prop',
    'starts_and_ends_once',  # TODO: if I recode this as 0 for no then-preference games, remove this from this list
)

EXPECTED_VALUE_0_FEATURES = (
    'adjacent_once_found', 'once_in_middle_of_pref_found', 'pref_without_hold_found',
    'repeated_variables_found', 'repeated_variable_type_in_either',
    'nested_logicals_found', 'identical_logical_children_found', 'identical_scoring_children_found',
    'tautological_expression_found',
    'redundant_expression_found', 'redundant_scoring_terminal_expression_found',
    'unnecessary_expression_found', 'unnecessary_scoring_terminal_expression_found',
    'total_score_non_positive', 'scoring_preferences_used_identically',
    'identical_consecutive_seq_func_predicates_found',
    'disjoint_preferences_scoring_terminal_types', 'disjoint_preferences_same_predicates_only',
    'disjoint_seq_funcs_found', 'predicate_without_variables_or_agent',
    'pref_forall_count_once_per_external_objects_used_incorrect',
    'pref_forall_external_forall_used_incorrect', 'pref_forall_used_incorrect',
    'pref_forall_correct_arity_incorrect', 'pref_forall_correct_types_incorrect',
    'two_number_operation_found', 'single_argument_multi_operation_found',
    'total_score_non_positive', 'scoring_preferences_used_identically',
    # 'section_without_pref_or_total_count_scoring',  # removed this feature
)

EXPECTED_FEATURE_VALUES = {
   feature: expected_value
   for feature_list, expected_value in zip((EXPECTED_VALUE_1_FEATURES, EXPECTED_VALUE_0_FEATURES), (1, 0))
   for feature in feature_list
}


class ExpectedFeatureValuesBehavioralFeature(FullFeaturesFitnessTerm):
    feature_to_value_dict: typing.Dict[str, int]
    game_result: bool
    def __init__(self, feature_to_value_dict: typing.Dict[str, int] = EXPECTED_FEATURE_VALUES, header: str = 'expected_feature_values'):
        super().__init__('', header)
        self.feature_to_value_dict = feature_to_value_dict
        self.game_result = False

    def game_start(self) -> None:
        self.game_result = False

    def update(self, ast, rule: str, context: ContextDict) -> None:
        pass

    def parse_full_features(self, features: typing.Dict[str, float]) -> None:
        self.game_result = all(value == int(features[feature]) for feature, value in self.feature_to_value_dict.items())

    def game_end(self):
        return self.game_result



PREDICATE_AND_OBJECT_GROUP_OBJECTS = [
    [room_and_object_types.BALLS, room_and_object_types.RECEPTACLES],
    [room_and_object_types.BLOCKS, room_and_object_types.BUILDING],
    [room_and_object_types.FURNITURE, room_and_object_types.ROOM_FEATURES],
    [room_and_object_types.SMALL_OBJECTS, room_and_object_types.LARGE_OBJECTS],
]

PREDICATE_AND_OBJECT_GROUP_OBJECTS_GAME_OBJECT = [
    [room_and_object_types.BALLS, room_and_object_types.RECEPTACLES],
    [room_and_object_types.BLOCKS, room_and_object_types.BUILDING],
    [room_and_object_types.FURNITURE, room_and_object_types.ROOM_FEATURES],
    [room_and_object_types.SMALL_OBJECTS, room_and_object_types.LARGE_OBJECTS],
    room_and_object_types.ANY_OBJECT,
]

PREDICATE_AND_OBJECT_GROUP_OBJECTS_BALL_BIN = [
    room_and_object_types.BALLS,
    room_and_object_types.RECEPTACLES,
    [room_and_object_types.BLOCKS, room_and_object_types.BUILDING],
    [room_and_object_types.FURNITURE, room_and_object_types.ROOM_FEATURES],
    [room_and_object_types.SMALL_OBJECTS, room_and_object_types.LARGE_OBJECTS],
]

PREDICATE_AND_OBJECT_GROUP_OBJECTS_BALL_BIN_GAME_OBJECT = [
    room_and_object_types.BALLS,
    room_and_object_types.RECEPTACLES,
    [room_and_object_types.BLOCKS, room_and_object_types.BUILDING],
    [room_and_object_types.FURNITURE, room_and_object_types.ROOM_FEATURES],
    [room_and_object_types.SMALL_OBJECTS, room_and_object_types.LARGE_OBJECTS],
    room_and_object_types.ANY_OBJECT,
]

PREDICATE_AND_OBJECT_GROUP_OBJECTS_EXPERIMENTAL_SMALLER = [
    room_and_object_types.BALLS,
    room_and_object_types.RECEPTACLES,
    [room_and_object_types.BLOCKS, room_and_object_types.BUILDING],
    [room_and_object_types.FURNITURE, room_and_object_types.ROOM_FEATURES],
    # [room_and_object_types.SMALL_OBJECTS, room_and_object_types.LARGE_OBJECTS],
    room_and_object_types.ANY_OBJECT,
]

PREDICATE_AND_OBJECT_GROUP_OBJECTS_EXPERIMENTAL_LARGER = [
    room_and_object_types.BALLS,
    room_and_object_types.RECEPTACLES,
    [room_and_object_types.BLOCKS, room_and_object_types.BUILDING],
    [room_and_object_types.FURNITURE, room_and_object_types.ROOM_FEATURES],
    [room_and_object_types.SMALL_OBJECTS, room_and_object_types.LARGE_OBJECTS],
    room_and_object_types.ANY_OBJECT,
]

PREDICATE_AND_OBJECT_GROUP_OBJECTS_FOR_EXEMPLAR_PREFERENCES = [
    room_and_object_types.BALLS,
    room_and_object_types.RECEPTACLES,
    [room_and_object_types.BLOCKS, room_and_object_types.BUILDING],
    [room_and_object_types.FURNITURE, room_and_object_types.ROOM_FEATURES],
    [room_and_object_types.SMALL_OBJECTS, room_and_object_types.LARGE_OBJECTS, room_and_object_types.ANY_OBJECT,],
]

PREDICATE_AND_OBJECT_GROUP_OBJECTS_WITH_PREF_COUNT = [
    [room_and_object_types.BALLS, room_and_object_types.RECEPTACLES],
    [room_and_object_types.BLOCKS, room_and_object_types.BUILDING],
    [room_and_object_types.FURNITURE, room_and_object_types.ROOM_FEATURES],
    [room_and_object_types.SMALL_OBJECTS, room_and_object_types.LARGE_OBJECTS, room_and_object_types.ANY_OBJECT,],
]


PREDICATE_AND_OBJECT_GROUP_PREDICATES = [
    ['agent_holds', 'in_motion'],
    'in',
    ['on', 'touch'],
    ['adjacent', 'between'],
]

PREDICATE_AND_OBJECT_GROUP_PREDICATES_EXPERIMENTAL = [
    ['agent_holds', 'in_motion'],
    ['in', 'on'],
    ['adjacent', 'near', 'touch'],
]

PREDICATE_AND_OBJECT_GROUP_PREDICATES_FOR_EXEMPLAR_PREFERENCES = [
    ['agent_holds', 'in_motion',],
    'in',
    'on',
    ['adjacent', 'near', 'touch'],
]

PREDICATE_AND_OBJECT_GROUP_PREDICATES_WITH_PREF_COUNT = [
    ['agent_holds', 'in_motion',],
    ['in', 'on',],
    ['adjacent', 'near', 'touch'],
]



class BehavioralFeatureSet(enum.Enum):
    BASIC_BINNED = 'basic_binned'
    BASIC_WITH_NODE_DEPTH = 'basic_with_node_depth'
    NODE_COUNT_OBJECTS = 'node_count_objects'
    NODE_COUNT_PREDICATES = 'node_count_predicates'
    NODE_COUNT_OBJECTS_SETUP = 'node_count_objects_setup'
    NODE_COUNT_PREDICATES_SETUP = 'node_count_predicates_setup'
    SPECIFIC_PREDICATES_SETUP = 'specific_predicates_setup'
    SPECIFIC_CATEGORIES_SETUP = 'specific_categories_setup'
    NODE_COUNT_SPECIFIC_PREDICATES = 'node_count_specific_predicates'
    PREDICATE_AND_OBJECT_GROUPS = 'predicate_and_object_groups'
    PREDICATE_AND_OBJECT_GROUPS_GAME_OBJECT = 'predicate_and_object_groups_go'
    PREDICATE_AND_OBJECT_GROUPS_SPLIT_BALL_BIN = 'predicate_and_object_groups_bb'
    PREDICATE_AND_OBJECT_GROUPS_SPLIT_BALL_BIN_GAME_OBJECT = 'predicate_and_object_groups_bb_go'
    PREDICATE_AND_OBJECT_GROUPS_FOR_EXEMPLAR_PREFERENCES = 'predicate_and_object_groups_exemplar_preferences'
    LATEST_WITH_AT_END = 'latest_at_end'
    LATEST_WITH_SETUP = 'latest_setup'
    LATEST_WITH_SETUP_AND_TERMINAL = 'latest_setup_terminal'
    LATEST_SETUP_EXPECTED_VALUES = 'latest_setup_expected_values'
    LATEST_AT_END_NO_GAME_OBJECT_SETUP = 'latest_at_end_no_game_object_setup'
    LATEST_AT_END_NO_GAME_OBJECT_EXPECTED_VALUES = 'latest_at_end_no_game_object_expected_values'
    EXEMPLAR_PREFERENCES_SETUP = 'exemplar_preferences_setup'
    EXEMPLAR_PREFERENCES_EXPECTED_VALUES = 'exemplar_preferences_expected_values'
    EXEMPLAR_PREFERENCES_BC_SETUP = 'exemplar_preferences_bc_setup'
    EXEMPLAR_PREFERENCES_BC_EXPECTED_VALUES = 'exemplar_preferences_bc_expected_values'
    EXEMPLAR_PREFERENCES_BC_NUM_PREFS_SETUP = 'exemplar_preferences_bc_num_prefs_setup'
    EXEMPLAR_PREFERENCES_BC_NUM_PREFS_EXPECTED_VALUES = 'exemplar_preferences_bc_num_prefs_expected_values'
    EXEMPLAR_PREFERENCES_BC_MAX_PREFS_SETUP = 'exemplar_preferences_bc_max_prefs_setup'
    EXEMPLAR_PREFERENCES_BC_MAX_PREFS_EXPECTED_VALUES = 'exemplar_preferences_bc_max_prefs_expected_values'
    PREDICATE_AND_OBJECT_GROUPS_SETUP_AT_END_PREF_COUNT_EXPECTED_VALUES = 'predicate_and_object_groups_setup_at_end_pref_count_expected_values'


class BehavioralFeaturizer(abc.ABC):
    @abc.abstractmethod
    def get_feature_names(self) -> typing.List[str]:
        pass

    @abc.abstractmethod
    def get_feature_value_counts(self) -> typing.Dict[str, int]:
        pass

    @abc.abstractmethod
    def get_game_features(self, game, features, partial_game: bool = False) -> typing.Dict[str, typing.Any]:
        pass


class FitnessFeaturesBehavioralFeaturizer(ASTFitnessFeaturizer, BehavioralFeaturizer):
    _feature_names_set: typing.Set[str]
    feature_value_counts_ignored_features: typing.List[str]
    full_features_registry: typing.Dict[str, FullFeaturesFitnessTerm]

    def __init__(self, args: argparse.Namespace,):
        super().__init__(args)
        self.full_features_registry = {}
        self.feature_value_counts_ignored_features = []
        self._feature_names_set = set()

    def register_full_features_term(self, term: FullFeaturesFitnessTerm, ignore_in_feature_value_counts: bool = True):
        self.full_features_registry[term.header] = term
        self.header_registry[term.header] = term
        self.headers.append(term.header)

        if ignore_in_feature_value_counts:
            self.feature_value_counts_ignored_features.append(term.header)

    def get_feature_names(self) -> typing.List[str]:
        #  [4:] to remove the first few automatically-added columns
        return self.get_all_column_keys()[4:]

    @property
    def feature_names_set(self):
        if len(self._feature_names_set) == 0:
            self._feature_names_set = set(self.get_feature_names())

        return self._feature_names_set

    def get_feature_value_counts(self) -> typing.Dict[str, int]:
        feature_to_term_mapping = self.get_column_to_term_mapping()
        n_values_by_feature = {}
        for feature_name in self.get_feature_names():
            if feature_name in self.feature_value_counts_ignored_features:
                continue

            feature_term = feature_to_term_mapping[feature_name]
            if hasattr(feature_term, 'get_n_values'):
                n_values = feature_term.get_n_values()  # type: ignore

            elif hasattr(feature_term, 'bins'):
                n_values = (len(feature_term.bins) + 1)  # type: ignore

            else:
                n_values = 2

            n_values_by_feature[feature_name] = n_values

        return n_values_by_feature

    def get_game_features(self, game, features, partial_game: bool = False, parse_context: typing.Optional[typing.Dict[str, typing.Any]] = None) -> typing.Dict[str, typing.Any]:
        behavioral_features = typing.cast(typing.Dict[str, float], self.parse(game, return_row=True, partial_ast=partial_game, additional_context=parse_context))
        behavioral_features = {key: val for key, val in behavioral_features.items() if key in self.feature_names_set}

        if len(self.full_features_registry) == 0:
            return behavioral_features  # type: ignore

        for header, term in self.full_features_registry.items():
            term.game_start()
            term.parse_full_features(features)
            term_result = term.game_end()

            if isinstance(term_result, bool):
                term_result = int(term_result)

            if isinstance(term_result, dict):
                for key, val in term_result.items():
                    header_key = f'{header}_{key}'
                    behavioral_features[header_key] = val
            else:
                behavioral_features[header] = term_result  # type: ignore

        return behavioral_features



DEFAULT_N_COMPONENTS = 32
DEFAULT_RANDOM_SEED = 33


class PCABehavioralFeaturizer(BehavioralFeaturizer):
    def __init__(self, feature_indices: typing.List[int], bins_per_feature: int,
                 ast_file_path: str, grammar_parser: tatsu.grammars.Grammar,  # type: ignore
                 fitness_featurizer: ASTFitnessFeaturizer, feature_names: typing.List[str],
                 n_components: int = DEFAULT_N_COMPONENTS, random_seed: int = DEFAULT_RANDOM_SEED,
                 should_init_pca: bool = True, should_init_features: bool = True, feature_prefix: str = 'pca'):

        self.feature_indices = feature_indices
        self.bins_per_feature = bins_per_feature
        self.ast_file_path = ast_file_path
        self.grammar_parser = grammar_parser
        self.fitness_featurizer = fitness_featurizer
        self.feature_names = feature_names
        self.n_components = n_components
        self.random_seed = random_seed
        self.feature_prefix = feature_prefix

        self.bins_by_feature_index = {}
        self.output_feature_names = [self._feature_name(i) for i in self.feature_indices]

        if should_init_pca:
            self._init_pca(should_init_features)

    def _game_to_feature_vector(self, game) -> np.ndarray:
        game_features = self.fitness_featurizer.parse(game, return_row=True)  # type: ignore
        return np.array([game_features[name] for name in self.feature_names])  # type: ignore

    def _init_pca(self, init_features: bool):
        game_asts = list(cached_load_and_parse_games_from_file(self.ast_file_path, self.grammar_parser, False))
        game_features = []
        for game in game_asts:
            game_features.append(self._game_to_feature_vector(game))

        features_array = np.stack(game_features)

        self.pca = PCA(n_components=self.n_components, random_state=self.random_seed)
        projections = self.pca.fit_transform(features_array)

        if init_features:
            for feature_index in self.feature_indices:
                feature_values = projections[:, feature_index]
                step = 1 / self.bins_per_feature
                quantiles = np.quantile(feature_values, np.linspace(step, 1 - step, self.bins_per_feature - 1))
                self.bins_by_feature_index[feature_index] = quantiles

                digits = np.digitize(feature_values, quantiles)
                counts = Counter(digits)
                logger.debug(f'On feature #{feature_index}, the real games have counts: {counts}')

            all_game_features = [self.get_game_features(game, None) for game in game_asts]
            all_game_feature_tuples = [tuple(game_features[name] for name in self.output_feature_names) for game_features in all_game_features]
            all_game_feature_tuples = set(all_game_feature_tuples)
            logger.debug(f'The real games have {len(all_game_feature_tuples)} unique feature tuples')

    def _feature_name(self, feature_index: int):
        return f'{self.feature_prefix}_{feature_index}'

    def get_feature_names(self) -> typing.List[str]:
        return self.output_feature_names

    def get_feature_value_counts(self) -> typing.Dict[str, int]:
        return {self._feature_name(i): self.bins_per_feature for i in self.feature_indices}

    def _project_game_pre_binning(self, game) -> np.ndarray:
        game_vector = self._game_to_feature_vector(game)
        return self.pca.transform(game_vector.reshape(1, -1))[0]

    def get_game_features(self, game, features, partial_game: bool = False) -> typing.Dict[str, typing.Any]:
        game_projection = self._project_game_pre_binning(game)
        return {self._feature_name(i): np.digitize(game_projection[i], self.bins_by_feature_index[i]) for i in self.feature_indices}


class ExemplarDistanceType(enum.Enum):
    FEATURE_VECTOR = 'feature_vector'
    PCA = 'pca'
    STRING_EDIT = 'string_edit'


class ExemplarDistanceMetric(enum.Enum):
    L1 = 'l1'
    L2 = 'l2'
    COSINE = 'cosine'

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self == ExemplarDistanceMetric.L1:
            return np.linalg.norm(a - b, ord=1)  # type: ignore
        elif self == ExemplarDistanceMetric.L2:
            return np.linalg.norm(a - b, ord=2)   # type: ignore
        elif self == ExemplarDistanceMetric.COSINE:
            return (1 - np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
        else:
            raise ValueError(f'Invalid distance metric: {self}')


class ExemplarDistanceFeaturizer(PCABehavioralFeaturizer):
    def __init__(self, distance_type: ExemplarDistanceType, distance_metric: ExemplarDistanceMetric,
                 feature_indices: typing.List[int], bins_per_feature: int,
                 ast_file_path: str, grammar_parser: tatsu.grammars.Grammar,  # type: ignore
                 fitness_featurizer: ASTFitnessFeaturizer, feature_names: typing.List[str],
                 n_components: int = DEFAULT_N_COMPONENTS, random_seed: int = DEFAULT_RANDOM_SEED):

        self.distance_type = distance_type
        self.distance_metric = distance_metric

        should_init_pca = distance_type == ExemplarDistanceType.PCA
        super().__init__(feature_indices, bins_per_feature, ast_file_path, grammar_parser, fitness_featurizer, feature_names, n_components, random_seed,
                         should_init_pca=should_init_pca, should_init_features=False, feature_prefix='exemplar_distance')

        game_asts = list(cached_load_and_parse_games_from_file(self.ast_file_path, self.grammar_parser, False))
        if self.distance_type == ExemplarDistanceType.STRING_EDIT:
            postprocessor = ast_parser.ASTSamplePostprocessor()
            game_strings = [ast_printer.ast_to_string(postprocessor(game), '\n') for game in game_asts]  # type: ignore

            self.exemplars = {i: game_strings[i] for i in feature_indices}
            exemplar_distances = {i: [edit_distance(game_strings[i], gs) for gs in game_strings] for i in feature_indices}

        elif self.distance_type == ExemplarDistanceType.FEATURE_VECTOR or self.distance_type == ExemplarDistanceType.PCA:
            game_features = []
            for game in game_asts:
                game_vector = self._game_to_feature_vector(game)
                if self.distance_type == ExemplarDistanceType.PCA:
                    game_vector = self.pca.transform(game_vector.reshape(1, -1))[0]

                game_features.append(game_vector)

            self.exemplars = {i: game_features[i] for i in feature_indices}
            exemplar_distances = {i: [self.distance_metric.distance(game_features[i], features) for features in game_features] for i in feature_indices}

        else:
            raise ValueError(f'Invalid distance type: {self.distance_type}')

        self.bins_by_feature_index = {}
        digits_by_feature_index = []
        for feature_index in self.feature_indices:
            feature_values = exemplar_distances[feature_index]
            step = 1 / self.bins_per_feature
            quantiles = np.quantile(feature_values, np.linspace(step, 1 - step, self.bins_per_feature - 1))
            self.bins_by_feature_index[feature_index] = quantiles

            digits = np.digitize(feature_values, quantiles)
            digits_by_feature_index.append(digits)
            counts = Counter(digits)
            logger.debug(f'On feature #{feature_index}, the real games have counts: {counts}')

        digit_tuples = list(zip(*digits_by_feature_index))
        unique_digit_tuples = set(digit_tuples)
        logger.debug(f'The real games have a total of {len(unique_digit_tuples)} unique digit tuples')


    def _project_game_pre_binning(self, game) -> typing.Union[np.ndarray, typing.Dict[int, float]]:
        if self.distance_type == ExemplarDistanceType.STRING_EDIT:
            game_string = ast_printer.ast_to_string(game, '\n')
            return {i: edit_distance(self.exemplars[i], game_string) for i in self.feature_indices}

        elif self.distance_type == ExemplarDistanceType.FEATURE_VECTOR:
            game_vector = self._game_to_feature_vector(game)
            return {i: self.distance_metric.distance(self.exemplars[i], game_vector) for i in self.feature_indices}  # type: ignore

        else:
            raise ValueError(f'Invalid distance type: {self.distance_type}')



EXEMPLAR_PREFERENCE_THRESHOLDS = {1: 0.15, 2: 0.3}
EXEMPLAR_PREFERENCE_IDS = [(5, 1), (8, 1), (48, 5), (66, 1), (74, 1)]


class ExemplarPreferenceDistanceFeaturizer(ast_parser.ASTParser, BehavioralFeaturizer):
    def __init__(self, ast_file_path: str,
                 grammar_parser: tatsu.grammars.Grammar,  # type: ignore
                 additional_featurizer: BehavioralFeaturizer,
                 thresholds: typing.Dict[int, float] = EXEMPLAR_PREFERENCE_THRESHOLDS,
                 exemplar_preference_ids: typing.List[typing.Tuple[int, int]] = EXEMPLAR_PREFERENCE_IDS):

        self.ast_file_path = ast_file_path
        self.grammar_parser = grammar_parser
        self.additional_featurizer = additional_featurizer
        self.thresholds = thresholds
        self.max_threshold = max(self.thresholds.values())
        self.exemplar_preference_ids = exemplar_preference_ids

        self.postprocessor = ast_parser.ASTSamplePostprocessor()

        self.exemplar_preference_indices = []
        self.all_preference_strings = []
        self.exemplar_index_quantiles = {}

        self.current_ast_preference_strings = []

        self._init_exemplars()

    def _init_exemplars(self):
        game_asts = list(cached_load_and_parse_games_from_file(self.ast_file_path, self.grammar_parser, False))

        exemplar_index = 0
        for i, game_ast in enumerate(game_asts):
            game_preferences = self(game_ast, should_postprocess=True)
            if exemplar_index < len(self.exemplar_preference_ids) and i == self.exemplar_preference_ids[exemplar_index][0]:
                self.exemplar_preference_indices.append(len(self.all_preference_strings) + self.exemplar_preference_ids[exemplar_index][1])
                exemplar_index += 1

            self.all_preference_strings.extend(game_preferences)

        pairwise_distances = np.zeros((len(self.all_preference_strings), len(self.all_preference_strings)))
        for i, j in itertools.combinations(range(len(self.all_preference_strings)), 2):
            pairwise_distances[i, j] = edit_distance(self.all_preference_strings[i], self.all_preference_strings[j])
            pairwise_distances[j, i] = pairwise_distances[i, j]

        for exemplar_index in self.exemplar_preference_indices:
            self.exemplar_index_quantiles[exemplar_index] = np.quantile(pairwise_distances[exemplar_index], np.linspace(0.01, 1, 100))

    def get_game_features(self, game, features, partial_game: bool = False, should_postprocess=False):
        game_features = self.additional_featurizer.get_game_features(game, features)

        game_preference_strings = self(game, should_postprocess=should_postprocess)
        all_distance_tuples = []
        for idx, pref_str in enumerate(game_preference_strings):
            for exemplar_idx in self.exemplar_preference_indices:
                d = edit_distance(pref_str, self.all_preference_strings[exemplar_idx])
                q = np.searchsorted(self.exemplar_index_quantiles[exemplar_idx], d) / 100
                all_distance_tuples.append((q, idx, exemplar_idx))

        all_distance_tuples.sort()
        exemplar_feature_values = {exemplar_idx: 0 for exemplar_idx in self.exemplar_preference_indices}
        used_preferences = set()
        used_exemplars = set()

        while all_distance_tuples:
            q, idx, exemplar_idx = all_distance_tuples.pop(0)

            if q > self.max_threshold:
                break

            if idx in used_preferences or exemplar_idx in used_exemplars:
                continue

            used_preferences.add(idx)
            used_exemplars.add(exemplar_idx)
            for feature_value, threshold in self.thresholds.items():
                if q <= threshold:
                    exemplar_feature_values[exemplar_idx] = feature_value
                    break

        for exemplar_idx in self.exemplar_preference_indices:
            game_features[f'exemplar_preference_{exemplar_idx}'] = exemplar_feature_values[exemplar_idx]

        return game_features

    def get_feature_names(self) -> typing.List[str]:
        feature_names = self.additional_featurizer.get_feature_names()
        for exemplar_idx in self.exemplar_preference_indices:
            feature_names.append(f'exemplar_preference_{exemplar_idx}')

        return feature_names

    def get_feature_value_counts(self) -> typing.Dict[str, int]:
        feature_value_counts = self.additional_featurizer.get_feature_value_counts()
        for exemplar_idx in self.exemplar_preference_indices:
            feature_value_counts[f'exemplar_preference_{exemplar_idx}'] = len(self.thresholds) + 1

        return feature_value_counts

    def __call__(self, ast: typing.Any, **kwargs) -> typing.List[str]:
        initial_call = 'inner_call' not in kwargs or not kwargs['inner_call']
        if initial_call:
            self.current_ast_preference_strings = []
            self._default_kwarg(kwargs, 'inner_call', True)
            self._default_kwarg(kwargs, 'should_postprocess', False)
            if kwargs['should_postprocess']:
                ast = self.postprocessor(ast)

        preference_node = isinstance(ast, tatsu.ast.AST) and ast.parseinfo.rule == 'preference'  # type: ignore

        if preference_node:
            self.current_ast_preference_strings.append(ast_printer.ast_section_to_string(ast.pref_body.body, ast_parser.PREFERENCES, '\n'))  # type: ignore
            inner_result = None

        else:
            inner_result = super().__call__(ast, **kwargs)

        if not initial_call:
            return inner_result  # type: ignore

        return self.current_ast_preference_strings


# BC_DISTANCE_EXEMPLAR_PREFERENCE_THRESHOLDS = {1: 1, 2: 2}
# BC_DISTANCE_EXEMPLAR_PREFERENCE_IDS = [(5, 1), (8, 2), (44, 1), (59, 0), (90, 2)]
# Trying a new set where I go for distance 1 or below
BC_DISTANCE_EXEMPLAR_PREFERENCE_THRESHOLDS = {1: 1}
# BC_DISTANCE_EXEMPLAR_PREFERENCE_IDS = [(5, 1), (11, 0), (17, 2), (44, 1), (48, 5), (49, 0), (51, 0), (67, 0), (69, 0)]
BC_DISTANCE_EXEMPLAR_PREFERENCE_IDS = [(5, 1), (11, 0), (17, 1), (26, 0), (48, 5), (49, 0), (49, 2), (64, 0), (69, 3)]


def dict_distance(d1, d2):
    return sum(int(d1[k] != d2[k]) for k in d1)


class ExemplarPreferenceBCDistanceFeaturizer(ast_parser.ASTParser, BehavioralFeaturizer):
    def __init__(self, preference_behavioral_featurizer: BehavioralFeaturizer,
                 ast_file_path: str,
                 grammar_parser: tatsu.grammars.Grammar,  # type: ignore
                 additional_features_featurizer: BehavioralFeaturizer,
                 thresholds: typing.Dict[int, int] = BC_DISTANCE_EXEMPLAR_PREFERENCE_THRESHOLDS,
                 exemplar_preference_ids: typing.List[typing.Tuple[int, int]] = BC_DISTANCE_EXEMPLAR_PREFERENCE_IDS,
                 count_total_matches: bool = False, max_match_count: int = MAX_NUM_PREFERENCES_COUNT,
                 reject_above_max_preference_count: bool = False, max_preference_count: int = MAX_NUM_PREFERENCES_COUNT
                 ):
        self.preference_behavioral_featurizer = preference_behavioral_featurizer

        self.ast_file_path = ast_file_path
        self.grammar_parser = grammar_parser
        self.additional_features_featurizer = additional_features_featurizer
        self.thresholds = thresholds
        self.max_threshold = max(self.thresholds.values())
        self.exemplar_preference_ids = exemplar_preference_ids
        self.count_total_matches = count_total_matches
        self.max_match_count = max_match_count
        self.reject_above_max_preference_count = reject_above_max_preference_count
        self.max_preference_count = max_preference_count

        self.postprocessor = ast_parser.ASTSamplePostprocessor()

        self.exemplar_preference_indices = []
        self.exemplar_features = {}
        self.exemplar_preference_asts = {}

        self.current_ast_preference_features = []

        self._init_exemplars()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'count_total_matches'):
            self.count_total_matches = False

        if not hasattr(self, 'max_match_count'):
            self.max_match_count = MAX_NUM_PREFERENCES_COUNT

        if not hasattr(self, 'reject_above_max_preference_count'):
            self.reject_above_max_preference_count = False

        if not hasattr(self, 'max_preference_count'):
            self.max_preference_count = MAX_NUM_PREFERENCES_COUNT

        if not hasattr(self, 'exemplar_preference_asts'):
            self.exemplar_preference_asts = {}

    def _init_exemplars(self):
        game_asts = list(cached_load_and_parse_games_from_file(self.ast_file_path, self.grammar_parser, False))
        all_preference_features = []
        all_preference_asts = []

        exemplar_index = 0
        for i, game_ast in enumerate(game_asts):
            game_preference_features, game_preference_asts = self(game_ast, should_postprocess=True, return_asts=True)

            while exemplar_index < len(self.exemplar_preference_ids) and i == self.exemplar_preference_ids[exemplar_index][0]:
                self.exemplar_preference_indices.append(len(all_preference_features) + self.exemplar_preference_ids[exemplar_index][1])
                exemplar_index += 1

            all_preference_features.extend(game_preference_features)
            all_preference_asts.extend(game_preference_asts)

        for exemplar_index in self.exemplar_preference_indices:
            self.exemplar_features[exemplar_index] = all_preference_features[exemplar_index]
            self.exemplar_preference_asts[exemplar_index] = all_preference_asts[exemplar_index]


    def get_game_features(self, game, features, partial_game: bool = False, should_postprocess=False):
        game_preference_features = self(game, should_postprocess=should_postprocess)

        if self.reject_above_max_preference_count and len(game_preference_features) > self.max_preference_count:
            raise SamplingException(f'Game has too many preferences: {len(game_preference_features)}')

        game_features = self.additional_features_featurizer.get_game_features(game, features)

        all_distance_tuples = []
        for idx, pref_features in enumerate(game_preference_features):
            for exemplar_idx in self.exemplar_preference_indices:
                d = dict_distance(pref_features, self.exemplar_features[exemplar_idx])
                all_distance_tuples.append((d, idx, exemplar_idx))

        all_distance_tuples.sort()
        exemplar_feature_values = {exemplar_idx: 0 for exemplar_idx in self.exemplar_preference_indices}
        used_preferences = set()
        count_by_exemplar = defaultdict(int)

        while all_distance_tuples:
            d, idx, exemplar_idx = all_distance_tuples.pop(0)

            if d > self.max_threshold:
                break

            if idx in used_preferences or (not self.count_total_matches and exemplar_idx in count_by_exemplar):
                continue

            used_preferences.add(idx)
            count_by_exemplar[exemplar_idx] += 1
            for feature_value, threshold in self.thresholds.items():
                if d <= threshold:
                    exemplar_feature_values[exemplar_idx] = feature_value
                    break

        for exemplar_idx in self.exemplar_preference_indices:
            if self.count_total_matches:
                game_features[f'exemplar_preference_{exemplar_idx}'] = min(self.max_match_count, count_by_exemplar[exemplar_idx])
            else:
                game_features[f'exemplar_preference_{exemplar_idx}'] = exemplar_feature_values[exemplar_idx]

        return game_features

    def get_feature_names(self) -> typing.List[str]:
        feature_names = self.additional_features_featurizer.get_feature_names()
        for exemplar_idx in self.exemplar_preference_indices:
            feature_names.append(f'exemplar_preference_{exemplar_idx}')

        return feature_names

    def get_feature_value_counts(self) -> typing.Dict[str, int]:
        feature_value_counts = self.additional_features_featurizer.get_feature_value_counts()
        count_per_exemplar =  self.max_match_count + 1 if self.count_total_matches else len(self.thresholds) + 1

        for exemplar_idx in self.exemplar_preference_indices:
            feature_value_counts[f'exemplar_preference_{exemplar_idx}'] = count_per_exemplar

        return feature_value_counts

    def __call__(self, ast: typing.Any, **kwargs) -> typing.List[typing.Dict[str, float]]:
        initial_call = 'inner_call' not in kwargs or not kwargs['inner_call']
        return_asts = initial_call and kwargs.get('return_asts', False)
        if initial_call:
            self.current_ast_preference_features = []
            self.current_preference_asts = []
            self._default_kwarg(kwargs, 'inner_call', True)

        preference_node = False

        if isinstance(ast, tatsu.ast.AST):
            rule = ast.parseinfo.rule  # type: ignore
            if rule == 'preference':
                preference_node = True

            elif rule == 'pref_forall':
                kwargs = ast_parser.update_context_variables(ast, kwargs)

        preference_node = isinstance(ast, tatsu.ast.AST) and ast.parseinfo.rule == 'preference'  # type: ignore

        if preference_node:
            context_vars = kwargs.get(ast_parser.VARIABLES_CONTEXT_KEY, {})
            pref_features = self.preference_behavioral_featurizer.get_game_features(
                ast, None, True,
                {
                    ast_parser.SECTION_CONTEXT_KEY: ast_parser.PREFERENCES,
                    ast_parser.VARIABLES_CONTEXT_KEY: context_vars,
                }  # type: ignore
            )
            self.current_ast_preference_features.append(pref_features)
            self.current_preference_asts.append(deepcopy_ast(ast, ASTCopyType.NODE))
            inner_result = None

        else:
            inner_result = super().__call__(ast, **kwargs)

        if not initial_call:
            return inner_result  # type: ignore

        if return_asts:
            return self.current_ast_preference_features, self.current_preference_asts  # type: ignore

        return self.current_ast_preference_features



def build_behavioral_features_featurizer(
        args: argparse.Namespace,
        grammar_parser: tatsu.grammars.Grammar,  # type: ignore
        fitness_featurizer: ASTFitnessFeaturizer, feature_names: typing.List[str]
        ) -> BehavioralFeaturizer:

    feature_set = args.map_elites_custom_behavioral_features_key

    if feature_set is not None:
        try:
            feature_set = BehavioralFeatureSet(feature_set)
        except ValueError:
            raise ValueError(f'Invalid feature set: {feature_set}')

        featurizer = FitnessFeaturesBehavioralFeaturizer(args)

        if feature_set == BehavioralFeatureSet.BASIC_BINNED:
            featurizer.register(NodeCount())
            featurizer.register(UniqueObjectsReferenced())
            featurizer.register(UniquePredicatesReferenced())

        elif feature_set == BehavioralFeatureSet.BASIC_WITH_NODE_DEPTH:
            featurizer.register(NodeCount())
            featurizer.register(UniqueObjectsReferenced())
            featurizer.register(UniquePredicatesReferenced())
            featurizer.register(MeanNodeDepth())

        elif feature_set == BehavioralFeatureSet.NODE_COUNT_OBJECTS:
            featurizer.register(NodeCount())
            featurizer.register(UniqueObjectsReferenced())

        elif feature_set == BehavioralFeatureSet.NODE_COUNT_PREDICATES:
            featurizer.register(NodeCount())
            featurizer.register(UniquePredicatesReferenced())

        elif feature_set == BehavioralFeatureSet.NODE_COUNT_OBJECTS_SETUP:
            featurizer.register(NodeCount())
            featurizer.register(UniqueObjectsReferenced())
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

        elif feature_set == BehavioralFeatureSet.NODE_COUNT_PREDICATES_SETUP:
            featurizer.register(NodeCount())
            featurizer.register(UniquePredicatesReferenced())
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

        elif feature_set == BehavioralFeatureSet.SPECIFIC_PREDICATES_SETUP:
            featurizer.register(PredicateUsed())
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

        elif feature_set == BehavioralFeatureSet.SPECIFIC_CATEGORIES_SETUP:
            featurizer.register(ObjectCategoryUsed())
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

        elif feature_set == BehavioralFeatureSet.NODE_COUNT_SPECIFIC_PREDICATES:
            featurizer.register(NodeCount(NODE_COUNT_BINS_8))
            featurizer.register(PredicateUsed())

        elif feature_set == BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS))

        elif feature_set == BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_GAME_OBJECT:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_GAME_OBJECT))

        elif feature_set == BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_SPLIT_BALL_BIN:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_BALL_BIN))

        elif feature_set == BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_SPLIT_BALL_BIN_GAME_OBJECT:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_BALL_BIN_GAME_OBJECT))

        elif feature_set == BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_FOR_EXEMPLAR_PREFERENCES:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES_FOR_EXEMPLAR_PREFERENCES))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_FOR_EXEMPLAR_PREFERENCES))
            featurizer.register(AtEndFound())

        elif feature_set == BehavioralFeatureSet.LATEST_WITH_AT_END:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES_EXPERIMENTAL))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_EXPERIMENTAL_LARGER))
            featurizer.register(AtEndFound())

        elif feature_set == BehavioralFeatureSet.LATEST_WITH_SETUP:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES_EXPERIMENTAL))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_EXPERIMENTAL_LARGER))
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

        elif feature_set == BehavioralFeatureSet.LATEST_WITH_SETUP_AND_TERMINAL:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES_EXPERIMENTAL))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_EXPERIMENTAL_SMALLER))
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP, ast_parser.TERMINAL]), section_rule=True)

        elif feature_set == BehavioralFeatureSet.LATEST_SETUP_EXPECTED_VALUES:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES_EXPERIMENTAL))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_EXPERIMENTAL_LARGER))
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)
            featurizer.register_full_features_term(ExpectedFeatureValuesBehavioralFeature())

        elif feature_set == BehavioralFeatureSet.LATEST_AT_END_NO_GAME_OBJECT_SETUP:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES_EXPERIMENTAL))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_FOR_EXEMPLAR_PREFERENCES))
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)
            featurizer.register(AtEndFound())

        elif feature_set == BehavioralFeatureSet.LATEST_AT_END_NO_GAME_OBJECT_EXPECTED_VALUES:
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES_EXPERIMENTAL))
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_FOR_EXEMPLAR_PREFERENCES))
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)
            featurizer.register(AtEndFound())
            featurizer.register_full_features_term(ExpectedFeatureValuesBehavioralFeature())

        elif feature_set == BehavioralFeatureSet.EXEMPLAR_PREFERENCES_SETUP:
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

            exemplar_preferences_featurizer = ExemplarPreferenceDistanceFeaturizer(
                args.map_elites_pca_behavioral_features_ast_file_path,
                grammar_parser,
                featurizer
            )

            featurizer = exemplar_preferences_featurizer

        elif feature_set == BehavioralFeatureSet.EXEMPLAR_PREFERENCES_EXPECTED_VALUES:
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)
            featurizer.register_full_features_term(ExpectedFeatureValuesBehavioralFeature())

            exemplar_preferences_featurizer = ExemplarPreferenceDistanceFeaturizer(
                args.map_elites_pca_behavioral_features_ast_file_path,
                grammar_parser,
                featurizer
            )

            featurizer = exemplar_preferences_featurizer

        elif feature_set == BehavioralFeatureSet.EXEMPLAR_PREFERENCES_BC_SETUP:
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)

            bc_featurizer_args = copy.deepcopy(args)
            bc_featurizer_args.map_elites_custom_behavioral_features_key = BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_FOR_EXEMPLAR_PREFERENCES.value
            preference_bc_featurizer = build_behavioral_features_featurizer(bc_featurizer_args, grammar_parser, fitness_featurizer, feature_names)

            exemplar_preferences_featurizer = ExemplarPreferenceBCDistanceFeaturizer(
                preference_bc_featurizer,
                args.map_elites_pca_behavioral_features_ast_file_path,
                grammar_parser,
                featurizer
            )

            featurizer = exemplar_preferences_featurizer

        elif feature_set == BehavioralFeatureSet.EXEMPLAR_PREFERENCES_BC_EXPECTED_VALUES:
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)
            featurizer.register_full_features_term(ExpectedFeatureValuesBehavioralFeature())

            bc_featurizer_args = copy.deepcopy(args)
            bc_featurizer_args.map_elites_custom_behavioral_features_key = BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_FOR_EXEMPLAR_PREFERENCES.value
            preference_bc_featurizer = build_behavioral_features_featurizer(bc_featurizer_args, grammar_parser, fitness_featurizer, feature_names)

            exemplar_preferences_featurizer = ExemplarPreferenceBCDistanceFeaturizer(
                preference_bc_featurizer,
                args.map_elites_pca_behavioral_features_ast_file_path,
                grammar_parser,
                featurizer
            )

            featurizer = exemplar_preferences_featurizer

        elif feature_set == BehavioralFeatureSet.EXEMPLAR_PREFERENCES_BC_NUM_PREFS_SETUP:
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)
            featurizer.register(NumPreferencesDefinedAsInteger())

            bc_featurizer_args = copy.deepcopy(args)
            bc_featurizer_args.map_elites_custom_behavioral_features_key = BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_FOR_EXEMPLAR_PREFERENCES.value
            preference_bc_featurizer = build_behavioral_features_featurizer(bc_featurizer_args, grammar_parser, fitness_featurizer, feature_names)

            exemplar_preferences_featurizer = ExemplarPreferenceBCDistanceFeaturizer(
                preference_bc_featurizer,
                args.map_elites_pca_behavioral_features_ast_file_path,
                grammar_parser,
                featurizer,
            )

            featurizer = exemplar_preferences_featurizer

        elif feature_set == BehavioralFeatureSet.EXEMPLAR_PREFERENCES_BC_NUM_PREFS_EXPECTED_VALUES:
            featurizer.register_full_features_term(ExpectedFeatureValuesBehavioralFeature())
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)
            featurizer.register(NumPreferencesDefinedAsInteger())

            bc_featurizer_args = copy.deepcopy(args)
            bc_featurizer_args.map_elites_custom_behavioral_features_key = BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_FOR_EXEMPLAR_PREFERENCES.value
            preference_bc_featurizer = build_behavioral_features_featurizer(bc_featurizer_args, grammar_parser, fitness_featurizer, feature_names)

            exemplar_preferences_featurizer = ExemplarPreferenceBCDistanceFeaturizer(
                preference_bc_featurizer,
                args.map_elites_pca_behavioral_features_ast_file_path,
                grammar_parser,
                featurizer,
            )

            featurizer = exemplar_preferences_featurizer

        elif feature_set == BehavioralFeatureSet.EXEMPLAR_PREFERENCES_BC_MAX_PREFS_SETUP:
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)
            featurizer.register(NumPreferencesDefinedAsInteger())

            bc_featurizer_args = copy.deepcopy(args)
            bc_featurizer_args.map_elites_custom_behavioral_features_key = BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_FOR_EXEMPLAR_PREFERENCES.value
            preference_bc_featurizer = build_behavioral_features_featurizer(bc_featurizer_args, grammar_parser, fitness_featurizer, feature_names)

            exemplar_preferences_featurizer = ExemplarPreferenceBCDistanceFeaturizer(
                preference_bc_featurizer,
                args.map_elites_pca_behavioral_features_ast_file_path,
                grammar_parser,
                featurizer,
                count_total_matches=True,
                reject_above_max_preference_count=True,
            )

            featurizer = exemplar_preferences_featurizer

        elif feature_set == BehavioralFeatureSet.EXEMPLAR_PREFERENCES_BC_MAX_PREFS_EXPECTED_VALUES:
            featurizer.register_full_features_term(ExpectedFeatureValuesBehavioralFeature())
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)
            featurizer.register(NumPreferencesDefinedAsInteger())

            bc_featurizer_args = copy.deepcopy(args)
            bc_featurizer_args.map_elites_custom_behavioral_features_key = BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_FOR_EXEMPLAR_PREFERENCES.value
            preference_bc_featurizer = build_behavioral_features_featurizer(bc_featurizer_args, grammar_parser, fitness_featurizer, feature_names)

            exemplar_preferences_featurizer = ExemplarPreferenceBCDistanceFeaturizer(
                preference_bc_featurizer,
                args.map_elites_pca_behavioral_features_ast_file_path,
                grammar_parser,
                featurizer,
                count_total_matches=True,
                reject_above_max_preference_count=True,
            )

            featurizer = exemplar_preferences_featurizer

        elif feature_set == BehavioralFeatureSet.PREDICATE_AND_OBJECT_GROUPS_SETUP_AT_END_PREF_COUNT_EXPECTED_VALUES:
            featurizer.register_full_features_term(ExpectedFeatureValuesBehavioralFeature())  # 1 bit
            featurizer.register(SectionExistsFitnessTerm([ast_parser.SETUP]), section_rule=True)  # 1 bit
            featurizer.register(NumPreferencesDefinedAsInteger())   # 2 bits
            featurizer.register(PredicateUsed(PREDICATE_AND_OBJECT_GROUP_PREDICATES_WITH_PREF_COUNT))  # 3 bits
            featurizer.register(ObjectCategoryUsed(PREDICATE_AND_OBJECT_GROUP_OBJECTS_WITH_PREF_COUNT))  # 4 bits
            featurizer.register(AtEndFound()) # 1 bit

        else:
            raise ValueError(f'Unimplemented feature set: {feature_set}')

        return featurizer

    indices = args.map_elites_pca_behavioral_features_indices
    bins_per_feature = args.map_elites_pca_behavioral_features_bins_per_feature

    if bins_per_feature is None:
        raise ValueError('Must specify bins per feature for PCA or Exempalr Distance featurizers')

    ast_file_path = args.map_elites_pca_behavioral_features_ast_file_path
    n_components = args.map_elites_pca_behavioral_features_n_components if args.map_elites_pca_behavioral_features_n_components is not None else max(indices) + 1
    random_seed = args.random_seed

    if args.map_elites_behavioral_feature_exemplar_distance_type is not None:
        if args.map_elites_behavioral_feature_exemplar_distance_metric is None:
            args.map_elites_behavioral_feature_exemplar_distance_metric = ExemplarDistanceMetric.L2.name

        exemplar_distance_featurizer = ExemplarDistanceFeaturizer(
            ExemplarDistanceType(args.map_elites_behavioral_feature_exemplar_distance_type.lower()),
            ExemplarDistanceMetric(args.map_elites_behavioral_feature_exemplar_distance_metric.lower()),
            feature_indices=indices,
            bins_per_feature=bins_per_feature,
            ast_file_path=ast_file_path,
            grammar_parser=grammar_parser,
            fitness_featurizer=fitness_featurizer,
            feature_names=feature_names,
            n_components=n_components,
            random_seed=random_seed
        )

        return exemplar_distance_featurizer

    pca_featurizer = PCABehavioralFeaturizer(
        feature_indices=indices,
        bins_per_feature=bins_per_feature,
        ast_file_path=ast_file_path,
        grammar_parser=grammar_parser,
        fitness_featurizer=fitness_featurizer,
        feature_names=feature_names,
        n_components=n_components,
        random_seed=random_seed
    )

    return pca_featurizer
