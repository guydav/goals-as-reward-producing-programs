from abc import ABC, abstractmethod
from bisect import bisect
from functools import reduce
import logging
import re
import typing

import numpy as np
import pandas as pd



class FitnessFeaturesPreprocessor(ABC):
    @abstractmethod
    def preprocess_df(self, df: pd.DataFrame, use_prior_values: bool = False,
                      min_max_values: typing.Optional[typing.Dict[str, typing.Tuple[float, float]]] = None) -> pd.DataFrame:
        pass


    @abstractmethod
    def preprocess_row(self, row: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        pass


NON_FEATURE_COLUMNS = set(['Index', 'src_file', 'game_name', 'domain_name', 'real', 'original_game_name'])

BINARIZE_IGNORE_FEATURES = set([
    'variables_defined_all', 'variables_used_all', 'preferences_used_all',
    'variables_defined_prop', 'variables_used_prop', 'preferences_used_prop',
    'setup_objects_used', 'setup_quantified_objects_used', 'any_setup_objects_used',
    'starts_and_ends_once', 'scoring_count_expression_repetitions_exist',
    'correct_predicate_function_arity', 'section_without_pref_or_total_count_terminal',
    'section_without_pref_or_total_count_scoring', 'adjacent_same_modal_found',
    'adjacent_once_found', 'once_in_middle_of_pref_found', 'pref_without_hold_found',
    'repeated_variables_found', 'repeated_variable_type_in_either',
    'nested_logicals_found', 'identical_logical_children_found',
    'identical_scoring_children_found', 'identical_scoring_expressions_found',
    'two_number_operation_found', 'single_argument_multi_operation_found', 'tautological_expression_found',
    'redundant_expression_found', 'redundant_scoring_terminal_expression_found',
    'unnecessary_expression_found', 'unnecessary_scoring_terminal_expression_found',
    'at_end_found', 'identical_consecutive_seq_func_predicates_found',
    'disjoint_preferences_found', 'disjoint_preferences_prop', 'disjoint_preferences_same_predicates_only',
    'disjoint_preferences_scoring_terminal_types', 'disjoint_preferences_scoring_terminal_predicates',
    'disjoint_seq_funcs_found', 'disjoint_at_end_found',
    'disjoint_modal_predicates_found', 'disjoint_modal_predicates_prop', 'predicate_without_variables_or_agent',
    'num_preferences_defined', 'total_score_non_positive', 'scoring_preferences_used_identically',
])

BINARIZE_IGNORE_PATTERNS = [
    re.compile(r'max_depth_[\w\d_]+'),
    re.compile(r'mean_depth_[\w\d_]+'),
    re.compile(r'node_count_[\w\d_]+'),
    re.compile(r'max_width_[\w\d_]+'),
    re.compile(r'pref_forall_[\w\d_]+_correct$'),
    re.compile(r'pref_forall_[\w\d_]+_incorrect$'),
    # re.compile(r'pref_forall_[\w\d_]+'),
    re.compile(r'length_of_then_modals_[\w\d_]+'),
    re.compile(r'max_quantification_count_[\w\d_]+'),
    re.compile(r'max_number_variables_types_quantified_[\w\d_]+'),
    re.compile(r'predicate_under_modal_[\w\d_]+'),
    re.compile(r'section_doesnt_exist_[\w\d_]+'),
    re.compile(r'num_preferences_defined_[\d_]+'),

    re.compile(r'predicate_found_in_data_[\w\d_]+'),
]

BINARIZE_NON_ONE = [
]

NGRAM_SCORE_PATTERN = re.compile(r'^(ast|text)_ngram(_\w+)?(_n_\d+)?_score$')
NGRAM_PATTERN = re.compile(r'^(ast|text)_ngram(_\w+)?(_n_\d+)?_\d+$')
ARG_TYPES_PATTERN = re.compile(r'[\w\d+_]+_arg_types[\w_]*')

SCALE_ZERO_ONE_PATTERNS = [
    NGRAM_SCORE_PATTERN,
    re.compile(r'[\w\d_]+_incorrect_count$'),  # since I now allow these to return a number, it might as well be scaled
    re.compile(r'scoring_count_expression_repetitions_max'),
]

BINRARIZE_NONZERO_PATTERNS = [
    ARG_TYPES_PATTERN,
    re.compile(r'compositionality_structure_\d+'),
    NGRAM_PATTERN,
]


DEFAULT_MISSING_VALUE_EPSILON = 1e-1


def set_missing_value_zero(series: typing.Optional[pd.Series], **kwargs):
    if series is None:
        return 0

    return series.fillna(0)


def set_missing_value_one(series: typing.Optional[pd.Series], **kwargs):
    if series is None:
        return 1

    return series.fillna(1)


def set_missing_value_min_epsilon(series: typing.Optional[pd.Series], min_value: typing.Optional[float] = None, epsilon: float = DEFAULT_MISSING_VALUE_EPSILON, **kwargs):
    if series is None:
        if min_value is None:
            raise ValueError('min_value must be provided if series is None')
        return min_value - epsilon

    # if series.name == 'ast_ngram_scoring_n_5_score':
    #     logging.debug(f'For series {series.name}, min_value is {min_value}, epsilon is {epsilon}')

    if min_value is None:
        min_value = np.nanmin(series.values)  # type: ignore
        # if series.name == 'ast_ngram_scoring_n_5_score':
        #     logging.debug(f'For series {series.name}, computed min_value {min_value}, min after subtracting epsilon {min_value - epsilon}')

    return series.fillna(min_value - epsilon), min_value  # type: ignore


COLUMN_NAME_OR_PATTERN_TO_MISSING_VALUE_FUNCTION = {
    ARG_TYPES_PATTERN: set_missing_value_zero,
    NGRAM_SCORE_PATTERN: set_missing_value_min_epsilon,
    NGRAM_PATTERN: set_missing_value_zero,
}


class BinarizeFitnessFeatures(FitnessFeaturesPreprocessor):
    columns: typing.List[str]
    ignore_columns: typing.Iterable[str]
    missing_value_epsilon: float
    missing_value_series_min_values: typing.Dict[str, float]
    scale_series_min_max_values: typing.Dict[str, typing.Tuple[float, float]]

    def __init__(self, ignore_columns: typing.Iterable[str] = NON_FEATURE_COLUMNS, missing_value_epsilon: float = DEFAULT_MISSING_VALUE_EPSILON):
        self.ignore_columns = ignore_columns
        self.missing_value_epsilon = missing_value_epsilon
        self.missing_value_series_min_values = {}
        self.scale_series_min_max_values = {}
        self.columns = []

    def _binarize_series(self, series: pd.Series, ignore_columns: typing.Iterable[str] = NON_FEATURE_COLUMNS, use_prior_values: bool = False):
        c = str(series.name)
        if c in ignore_columns:
            return series

        if c in BINARIZE_IGNORE_FEATURES:
            return series

        if any([p.match(c) for p in BINARIZE_IGNORE_PATTERNS]):
            return series

        if c in BINARIZE_NON_ONE:
            return (series == 1).astype(int)

        if any([p.match(c) for p in SCALE_ZERO_ONE_PATTERNS]):
            if use_prior_values:
                min_val, max_val = self.scale_series_min_max_values[c]

            else:
                min_val, max_val = series.min(), series.max()
                self.scale_series_min_max_values[c] = (min_val, max_val)

            if min_val == max_val:
                return np.clip(series, 0, 1)

            return np.clip((series - min_val) / (max_val - min_val), 0, 1)

        if any([p.match(c) for p in BINRARIZE_NONZERO_PATTERNS]):
            return (series != 0).astype(int)

        raise ValueError(f'No binarization rule for column {c}')

    def _fill_series_missing_values(self, series: pd.Series, use_prior_values: bool = False,
                                    min_max_values: typing.Optional[typing.Dict[str, typing.Tuple[float, float]]] = None):
        if not series.isna().any():
            return series

        # logging.info(f'Filling missing values for column {series.name}')
        c = str(series.name)
        missing_value_function = None
        if c in COLUMN_NAME_OR_PATTERN_TO_MISSING_VALUE_FUNCTION:
            missing_value_function = COLUMN_NAME_OR_PATTERN_TO_MISSING_VALUE_FUNCTION[c]

        else:
            for p, f in COLUMN_NAME_OR_PATTERN_TO_MISSING_VALUE_FUNCTION.items():
                if isinstance(p, re.Pattern) and p.match(c):
                    missing_value_function = f
                    break

        # if series.name == 'ast_ngram_scoring_n_5_score':
        #     logging.debug(f'ast_ngram_scoring_n_5_score: {series.isna().sum()} missing values, {series.isna().sum() / 1025}, found {missing_value_function}')

        if missing_value_function is None:
            raise ValueError(f'No missing value function for column {c}')

        if use_prior_values and c in self.missing_value_series_min_values:
            result = missing_value_function(series, min_value=self.missing_value_series_min_values[c], epsilon=self.missing_value_epsilon)
        else:
            if min_max_values is not None and c in min_max_values:
                result = missing_value_function(series, epsilon=self.missing_value_epsilon,
                                                min_value=min_max_values[c][0],
                                                max_value=min_max_values[c][1])
            else:
                result = missing_value_function(series, epsilon=self.missing_value_epsilon)

        if isinstance(result, tuple):
            series, min_value = result
            self.missing_value_series_min_values[c] = min_value
            return series

        else:
            return result

    def _preprocess_series(self, series: pd.Series, use_prior_values: bool = False,
                           min_max_values: typing.Optional[typing.Dict[str, typing.Tuple[float, float]]] = None):
        # if series.name == 'ast_ngram_scoring_n_5_score':
        #     logging.debug(f'ast_ngram_scoring_n_5_score: {series.isna().sum()} missing values, {series.isna().sum() / 1025}')
        series = self._fill_series_missing_values(series, use_prior_values=use_prior_values, min_max_values=min_max_values)
        # if series.name == 'ast_ngram_scoring_n_5_score':
        #     logging.debug(f'ast_ngram_scoring_n_5_score, min after fill: {series.min()} ({(series == series.min()).sum() / 1025:.2f}%)')
        series = self._binarize_series(series, use_prior_values=use_prior_values, ignore_columns=self.ignore_columns)
        # if series.name == 'ast_ngram_scoring_n_5_score':
        #     logging.debug(f'ast_ngram_scoring_n_5_score, after binarizing: {series.min()} ({(series == series.min()).sum() / 1025:.2f}%)')
        return series

    def preprocess_df(self, df: pd.DataFrame, use_prior_values: bool = False,
                      min_max_values: typing.Optional[typing.Dict[str, typing.Tuple[float, float]]] = None) -> pd.DataFrame:
        if not use_prior_values:
            self.columns = [str(c) for c in df.columns]

        if use_prior_values and not self.columns:
            raise ValueError('Must call preprocess_df without use_prior_values before preprocess_df with use_prior_values')

        return df.apply(self._preprocess_series, axis=0, use_prior_values=use_prior_values, min_max_values=min_max_values)

    def preprocess_row(self, row: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        if not self.columns:
            raise ValueError('Must call preprocess_df before preprocess_row')

        for c in self.columns:
            if c not in row or row[c] is None:
                missing_value_function = None
                if c in COLUMN_NAME_OR_PATTERN_TO_MISSING_VALUE_FUNCTION:
                    missing_value_function = COLUMN_NAME_OR_PATTERN_TO_MISSING_VALUE_FUNCTION[c]

                else:
                    for p, f in COLUMN_NAME_OR_PATTERN_TO_MISSING_VALUE_FUNCTION.items():
                        if isinstance(p, re.Pattern) and p.match(c):
                            missing_value_function = f
                            break

                if missing_value_function is None:
                    raise ValueError(f'No missing value function for column {c} with no value in the row')

                if self.missing_value_series_min_values and c in self.missing_value_series_min_values:
                    row[c] = missing_value_function(None, min_value=self.missing_value_series_min_values[c], epsilon=self.missing_value_epsilon)
                else:
                    row[c] = missing_value_function(None, epsilon=self.missing_value_epsilon)

            v = row[c]

            if c in BINARIZE_NON_ONE:
                row[c] = 1 if v == 1 else 0

            elif c in self.scale_series_min_max_values:
                min_val, max_val = self.scale_series_min_max_values[c]
                row[c] = np.clip((v - min_val) / (max_val - min_val), 0, 1)

            elif any([p.match(c) for p in BINRARIZE_NONZERO_PATTERNS]):
                row[c] = 1 if v != 0 else 0

        return row


DEFAULT_MERGE_THRESHOLD_PROPORTION = 0.01   # ~ 1000 games out of the 98 * 1025
DEFAULT_FEATURE_SUFFIXES = ('setup', 'constraints')
DEFAULT_MERGE_COLUMN_SUFFIX = 'other'


# def _merge_single_prefix(df: pd.DataFrame, feature_prefix: str, threshold: int = DEFAULT_MERGE_THRESHOLD,
#     merge_function: typing.Callable = np.logical_or, merged_column_suffix: str = DEFAULT_MERGE_COLUMN_SUFFIX,
#     feature_suffix: str = '') -> None:

#     merged_column_key = f'{feature_prefix}_{merged_column_suffix}{"_" + feature_suffix if feature_suffix else ""}'
#     prefix_feature_names = [c for c in df.columns if c.startswith(feature_prefix) and c.endswith(feature_suffix)]
#     if len(prefix_feature_names) == 0:
#         raise ValueError(f'No features found for prefix {feature_prefix} and suffix {feature_suffix}')
#     merged_column_insert_index = bisect(prefix_feature_names, merged_column_key)
#     first_prefix_feature_index = list(df.columns).index(prefix_feature_names[0])
#     insert_index = first_prefix_feature_index + merged_column_insert_index

#     counts = df[[c for c in df.columns if c.startswith(feature_prefix) and c.endswith(feature_suffix)]].sum()
#     keys_to_merge = counts.index[counts < threshold]  # type: ignore
#     if len(keys_to_merge) == 0:
#         print(feature_prefix)
#         return
#     new_series_values = reduce(merge_function, [df[k] for k in keys_to_merge[1:]], df[keys_to_merge[0]]).astype(int)

#     df.insert(insert_index, merged_column_key, new_series_values)
#     df.drop(keys_to_merge, axis=1, inplace=True)


# def merge_sparse_features(df: pd.DataFrame, predicates: typing.Sequence[str],
#     threshold: int = DEFAULT_MERGE_THRESHOLD, merge_function: typing.Callable = np.logical_or,
#     merged_column_suffix: str = DEFAULT_MERGE_COLUMN_SUFFIX, feature_suffixes: typing.Sequence[str] = DEFAULT_FEATURE_SUFFIXES
#     ) -> pd.DataFrame:

#     df = df.copy(deep=True)

#     for feature_suffix in feature_suffixes:
#         for p in predicates:
#             feature_prefix = f'{p}_arg_types'
#             _merge_single_prefix(df, feature_prefix, threshold, merge_function, merged_column_suffix, feature_suffix)

#             # if p not in PREDICATE_FUNCTION_ARITY_MAP:
#             #     raise ValueError(f'Predicate {p} not in arity map')

#             # arity = PREDICATE_FUNCTION_ARITY_MAP[p]
#             # if arity == 1:
#             #     feature_prefix = f'arg_types_{p}'
#             #     _merge_single_prefix(df, feature_prefix, threshold, merge_function, merged_column_suffix, feature_suffix)

#             # else:  # arity = 2/3
#             #     for c in CATEGORIES_TO_TYPES.keys():
#             #         if c == EMPTY_OBJECT:
#             #             continue
#             #         feature_prefix = f'arg_types_{p}_{c}'
#             #         _merge_single_prefix(df, feature_prefix, threshold, merge_function, merged_column_suffix, feature_suffix)

    # return df


class MergeFitnessFeatures(FitnessFeaturesPreprocessor):
    df_key_to_index: typing.Dict[str, int]
    dropped_keys: typing.Set[str]
    feature_suffixes: typing.Sequence[str]
    forced_output_keys: typing.Optional[typing.Set[str]]
    keys_to_drop: typing.List[str]
    merge_function: typing.Callable
    merged_column_suffix: str
    merged_key_indices: typing.Dict[str, int]
    merged_key_to_original_keys: typing.Dict[str, typing.List[str]]
    predicates: typing.Sequence[str]
    threshold_proportion: float

    def __init__(self, predicates: typing.Sequence[str], threshold_proportion: float = DEFAULT_MERGE_THRESHOLD_PROPORTION,
                 merge_function: typing.Callable = np.logical_or, merged_column_suffix: str = DEFAULT_MERGE_COLUMN_SUFFIX,
                 feature_suffixes: typing.Sequence[str] = DEFAULT_FEATURE_SUFFIXES, default_value: int = 0,
                 forced_output_keys: typing.Optional[typing.Set[str]] = None):

        self.predicates = predicates
        self.threshold_proportion = threshold_proportion
        self.merge_function = merge_function
        self.merged_column_suffix = merged_column_suffix
        self.feature_suffixes = feature_suffixes
        self.default_value = default_value
        self.forced_output_keys = None

        self.dropped_keys = set()
        self.keys_to_drop = []
        self.merged_key_indices = {}
        self.merged_key_to_original_keys = {}

    def _merge_single_prefix(self, df: pd.DataFrame, feature_prefix: str, feature_suffix: str = '') -> None:

        merged_column_key = f'{feature_prefix}_{self.merged_column_suffix}{"_" + feature_suffix if feature_suffix else ""}'
        prefix_feature_names = [str(c) for c in df.columns if str(c).startswith(feature_prefix) and str(c).endswith(feature_suffix)]
        if len(prefix_feature_names) == 0:
            logging.info(f'No features found for prefix {feature_prefix} and suffix {feature_suffix}')
            prefix_feature_names = [c for c in df.columns if str(c).startswith(feature_prefix)]
            if len(prefix_feature_names) > 0:
                last_prefix_feature = prefix_feature_names[-1]
                insert_index = df.columns.get_loc(last_prefix_feature) + 1

            else:
                all_arg_type_columns = [c for c in df.columns if 'arg_types' in str(c)]
                insert_index = df.columns.get_loc(all_arg_type_columns[-1]) + 1

            new_series_values = pd.Series(np.ones(df.shape[0]) * self.default_value, name=merged_column_key)
            keys_to_merge = []

        else:
            merged_column_insert_index = bisect(prefix_feature_names, merged_column_key)
            if merged_column_insert_index >= len(prefix_feature_names):
                merge_insert_feature_name = prefix_feature_names[-1]
                insert_index = df.columns.get_loc(merge_insert_feature_name) + 1

            else:
                merge_insert_feature_name = prefix_feature_names[merged_column_insert_index]
                insert_index = df.columns.get_loc(merge_insert_feature_name)

            counts = df[prefix_feature_names].sum()
            keys_to_merge = list(counts.index[counts < self.threshold_proportion * df.shape[0]])  # type: ignore

            if self.forced_output_keys is not None:
                fixed_keys_to_merge = []
                for k in keys_to_merge:
                    if k not in self.forced_output_keys:
                        fixed_keys_to_merge.append(k)

                for k in prefix_feature_names:
                    if k not in self.forced_output_keys and k not in fixed_keys_to_merge:
                        fixed_keys_to_merge.append(k)

                keys_to_merge = fixed_keys_to_merge

            if len(keys_to_merge) == 0:
                logging.info(f'No features to merge for prefix {feature_prefix} and suffix {feature_suffix}')
                return

            # print(f'Merging {len(keys_to_merge)} features for prefix {feature_prefix} and suffix {feature_suffix}')
            new_series_values = reduce(self.merge_function, [df[k] for k in keys_to_merge[1:]], df[keys_to_merge[0]]).astype(int)


        df.insert(insert_index, merged_column_key, new_series_values)
        self.keys_to_drop.extend(keys_to_merge)  # type: ignore

        self.merged_key_indices[merged_column_key] = insert_index
        self.merged_key_to_original_keys[merged_column_key] = list(keys_to_merge)  # type: ignore

    def _merge_single_prefix_from_prior_merge(self, df: pd.DataFrame, feature_prefix: str, feature_suffix: str = '') -> None:
        merged_column_key = f'{feature_prefix}_{self.merged_column_suffix}{"_" + feature_suffix if feature_suffix else ""}'
        if merged_column_key in self.merged_key_indices:
            insert_index = self.merged_key_indices[merged_column_key]
            keys_to_merge = self.merged_key_to_original_keys[merged_column_key]
            keys_to_merge = [k for k in keys_to_merge if k in df.columns]
            new_series_values = reduce(self.merge_function, [df[k] for k in keys_to_merge[1:]], df[keys_to_merge[0]]).astype(int)
            df.insert(insert_index, merged_column_key, new_series_values)

    def preprocess_df(self, df: pd.DataFrame, use_prior_values: bool = False,
                      min_max_values: typing.Optional[typing.Dict[str, typing.Tuple[float, float]]] = None) -> pd.DataFrame:
        if not use_prior_values:
            self.keys_to_drop = []

        df = df.copy(deep=True)

        for feature_suffix in self.feature_suffixes:
            for p in self.predicates:
                feature_prefix = f'{p}_arg_types'
                if use_prior_values:
                    self._merge_single_prefix_from_prior_merge(df, feature_prefix, feature_suffix)
                else:
                    self._merge_single_prefix(df, feature_prefix, feature_suffix)

        df.drop(self.keys_to_drop, axis=1, inplace=True)
        self.dropped_keys = set(self.keys_to_drop)

        return df

    def preprocess_row(self, row: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        # TODO: this might be slow because of the number of inserts, consider optimizing further if it proves annoying
        keys = list(row.keys())
        merged_key_values = {}

        for merged_key in self.merged_key_to_original_keys:
            insert_index = self.merged_key_indices[merged_key]
            keys.insert(insert_index, merged_key)
            merged_keys = self.merged_key_to_original_keys[merged_key]
            if len(merged_keys) == 0:
                merged_key_values[merged_key] = self.default_value

            else:
                first_key = merged_keys[0]
                merged_key_values[merged_key] = int(reduce(self.merge_function,
                    [row[k] if k in row else self.default_value for k in self.merged_key_to_original_keys[merged_key]],
                    row[first_key] if first_key in row else self.default_value))

        new_row = {}
        for k in keys:
            if k in self.dropped_keys:
                continue
            if k in merged_key_values:
                new_row[k] = merged_key_values[k]
            else:
                new_row[k] = row[k]

        return new_row
