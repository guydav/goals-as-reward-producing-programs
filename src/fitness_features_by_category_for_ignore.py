from dataclasses import dataclass
import re

from fitness_features import COMMON_SENSE_PREDICATES_FUNCTIONS
from fitness_features_preprocessing import NGRAM_SCORE_PATTERN, ARG_TYPES_PATTERN

GRAMMAR_CONTEXT_FEATURES = [
    'variables_used_all',
    'variables_used_prop',
    'preferences_used_all',
    'preferences_used_prop',

    'repeated_variables_found',
    'repeated_variable_type_in_either',

    'section_doesnt_exist_setup',
    'section_doesnt_exist_terminal',
]

NGRAM_AND_DATA_BASED_FEATURES = [
    NGRAM_SCORE_PATTERN,
    re.compile(r'predicate_found_in_data_[\w\d_]+'),
]

COUNTING_FEATURES = [
    # How many preferences are defined
    re.compile(r'num_preferences_defined_[\d_]+'),
    # How many modals are under a then
    re.compile(r'length_of_then_modals_[\w\d_]+'),
    # Various features related to variable quantifications
    re.compile(r'max_quantification_count_[\w\d_]+'),
    re.compile(r'max_number_variables_types_quantified_[\w\d_]+'),
    # Man and max depth and node count
    re.compile(r'max_depth_[\w\d_]+'),
    re.compile(r'mean_depth_[\w\d_]+'),
    re.compile(r'node_count_[\w\d_]+'),
    re.compile(r'max_width_[\w\d_]+'),
]

# A Smaller set of counting features to exclude
COUNTING_LESS_IMPORTANT_FEATURES = [
    # 2023-11-23: trying without the feature for how many preferences are defined, appears to not be great
    # How many preferences are defined
    re.compile(r'num_preferences_defined_[\d_]+'),
    # How many modals are under a then
    re.compile(r'length_of_then_modals_[\w\d_]+'),
    # Various features related to variable quantifications
    re.compile(r'max_quantification_count_[\w\d_]+'),
    re.compile(r'max_number_variables_types_quantified_[\w\d_]+'),  # CONSIDER-RESTORING
    # Man and max depth and node count -- keep max depth and node count
    # 2023-11-13 -- trying with the max width, too, so only mean depth removed
    # 2023-11-20 -- actually, max width seems to induce weird behavior
    # 2023-11-23 -- trying with max width one more time after some more recent results
    # 2023-11-28 -- and without, as it's still a little questionable if it helps or hurts more
    # re.compile(r'node_count_[\w\d_]+'),
    # re.compile(r'max_depth_[\w\d_]+'),
    re.compile(r'mean_depth_[\w\d_]+'),
    re.compile(r'max_width_[\w\d_]+'),  # CONSIDER-RESTORING  # CONSIDER-REMOVING
]

COUNTING_FEATURES_PATTERN_DICT = {
    # How many preferences are defined
    'num_preferences_defined': re.compile(r'num_preferences_defined_[\d_]+'),
    # How many modals are under a then
    'modals under `then`': re.compile(r'length_of_then_modals_[\w\d_]+'),
    # Various features related to variable quantifications
    'variables quantified': re.compile(r'max_quantification_count_[\w\d_]+'),
    'variable types quantified': re.compile(r'max_number_variables_types_quantified_[\w\d_]+'),
    # Man and max depth and node count
    'max depth by section': re.compile(r'max_depth_[\w\d_]+'),
    'mean depth by section': re.compile(r'mean_depth_[\w\d_]+'),
    'node count by section': re.compile(r'node_count_[\w\d_]+'),
    'max width by section': re.compile(r'max_width_[\w\d_]+'),
}


FORALL_FEATURES = [
    re.compile(r'pref_forall_[\w\d_]+_correct$'),
    re.compile(r'pref_forall_[\w\d_]+_incorrect$'),
    re.compile(r'[\w\d_]+_incorrect_count$'),
]


FORALL_LESS_IMPORTANT_FEATURES = [
    # re.compile(r'pref_forall_[\w\d_]+_correct$'),
    # re.compile(r'pref_forall_[\w\d_]+_incorrect$'),
    re.compile(r'[\w\d_]+_incorrect_count$'),
]


PREDICATE_UNDER_MODAL_FEATURES = [
    re.compile(r'predicate_under_modal_[\w\d_]+'),
]

PREDICATE_ROLE_FILLER_FEATURES = [
    ARG_TYPES_PATTERN
]

PREDICATE_ROLE_FILLER_PATTERN_DICT = {
    pred: re.compile(f'{pred}_arg_types_[\\w_]+')
    for pred in COMMON_SENSE_PREDICATES_FUNCTIONS
}

COMPOSITIONALITY_FEATURES = [
    re.compile(r'compositionality_structure_\d+'),
]

GRAMMAR_USE_FEATURES = [
    'variables_defined_all',
    'variables_defined_prop',

    'setup_objects_used',
    'setup_quantified_objects_used',
    'any_setup_objects_used',
    'adjacent_once_found',
    'adjacent_same_modal_found',
    'starts_and_ends_once',
    'once_in_middle_of_pref_found',
    'pref_without_hold_found',
    'at_end_found',

    'nested_logicals_found',
    'identical_logical_children_found',
    'identical_scoring_children_found',
    'scoring_count_expression_repetitions_exist',
    'tautological_expression_found',
    'redundant_expression_found',
    'redundant_scoring_terminal_expression_found',
    'unnecessary_expression_found',
    'unnecessary_scoring_terminal_expression_found',
    'identical_consecutive_seq_func_predicates_found',
    'disjoint_preferences_found',
    'disjoint_preferences_prop',
    'disjoint_preferences_scoring_terminal_types',
    'disjoint_preferences_scoring_terminal_predicates',
    'disjoint_preferences_same_predicates_only',
    'disjoint_seq_funcs_found',
    'disjoint_at_end_found',
    'disjoint_modal_predicates_found',
    'disjoint_modal_predicates_prop',
    'predicate_without_variables_or_agent',

    'two_number_operation_found',
    'single_argument_multi_operation_found',

    'section_without_pref_or_total_count_terminal',
    'section_without_pref_or_total_count_scoring',

    'total_score_non_positive',
    'scoring_preferences_used_identically'
]

# The ones that are kept are the ones commented out
GRAMMAR_USE_LESS_IMPORTANT_FEATURES = [
    # Trying without these, since the minor context-fixing we do now handles this
    'variables_defined_all',
    'variables_defined_prop',

    # 2023-11-23: trying without the feature for any setup objects used
    'setup_objects_used',
    # 2023-09-21: trying without the specific feature to the quantified objects
    # 2023-11-22: trying with it again
    # 'setup_quantified_objects_used',
    # 'any_setup_objects_used,
    # 'adjacent_once_found',
    # 'adjacent_same_modal_found',
    'starts_and_ends_once',
    # 2023-11-20: trying with this feature
    # 'once_in_middle_of_pref_found',
    # 'pref_without_hold_found',  # CONSIDER-RESTORING
    # 2023-09-21: trying without an explicit marking of at_end
    'at_end_found',

    # 'nested_logicals_found',  # CONSIDER-RESTORING
    # 'identical_logical_children_found',  # CONSIDER-RESTORING
    # 'identical_scoring_children_found',  # CONSIDER-RESTORING
    'scoring_count_expression_repetitions_exist',
    'tautological_expression_found',  # CONSIDER-RESTORING
    # 'redundant_expression_found',
    # 'redundant_scoring_terminal_expression_found',
    # 'unnecessary_expression_found',
    # 'unnecessary_scoring_terminal_expression_found',
    # 'identical_consecutive_seq_func_predicates_found',
    # 'disjoint_preferences_found',
    'disjoint_preferences_prop',   #  CONSIDER-REMOVING
    # 'disjoint_preferences_scoring_terminal_types',
    # 'disjoint_preferences_scoring_terminal_predicates', # CONSIDER-REMOVING
    'disjoint_preferences_same_predicates_only',  # CONSIDER-REMOVING
    # 'disjoint_seq_funcs_found',
    # 'disjoint_at_end_found',
    # 'disjoint_modal_predicates_found',  # CONSIDER-REMOVING
    # 'disjoint_modal_predicates_prop',
    # 'predicate_without_variables_or_agent',

    # 'two_number_operation_found',
    'single_argument_multi_operation_found',

    'section_without_pref_or_total_count_terminal', # CONSIDER-REMOVING
    'section_without_pref_or_total_count_scoring',  # CONSIDER-REMOVING

    # 'total_score_non_positive',
    # 'scoring_preferences_used_identically'
]

FEATURE_CATEGORIES = {
    'grammar_context': GRAMMAR_CONTEXT_FEATURES,
    'ngram_and_data_based': NGRAM_AND_DATA_BASED_FEATURES,
    'counting': COUNTING_FEATURES,
    'counting_less_important': COUNTING_LESS_IMPORTANT_FEATURES,
    'forall': FORALL_FEATURES,
    'forall_less_important': FORALL_LESS_IMPORTANT_FEATURES,
    'predicate_under_modal': PREDICATE_UNDER_MODAL_FEATURES,
    'predicate_role_filler': PREDICATE_ROLE_FILLER_FEATURES,
    'compositionality': COMPOSITIONALITY_FEATURES,
    'grammar_use': GRAMMAR_USE_FEATURES,
    'grammar_use_less_important': GRAMMAR_USE_LESS_IMPORTANT_FEATURES,
}



@dataclass
class FeatureDescription:
    name: str
    description: str
    type: str

    def __str__(self):
        return f'{self.name} ({self.type}): {self.description}'

    def to_latex(self):
        name = self.name.replace('_', '\\_')
        desc = self.description.replace('_', '\\_')
        return f'\\item \\texttt{{{name} [{self.type}]}}: {desc}'

BINARY_FEATURE = 'b'
FLOAT_FEATURE = 'f'
DISCRETIZED_FEATURE = 'd'
PROPORTION_FEATURE = 'p'


FITNESS_FEATURE_DESCRIPTIONS = [
    FeatureDescription('ast_ngram_full_n_5_score', 'What is the mean 5-gram model score under an n-gram model trained on the real games?', FLOAT_FEATURE),
    FeatureDescription('ast_ngram_setup_n_5_score', 'What is the mean 5-gram model score of the setup section under an n-gram model trained on the real game setup sections?', FLOAT_FEATURE),
    FeatureDescription('ast_ngram_constraints_n_5_score', 'What is the mean 5-gram model score of the gameplay preferences section under an n-gram model trained on the real game preferences sections?', FLOAT_FEATURE),
    FeatureDescription('ast_ngram_terminal_n_5_score', 'What is the mean 5-gram model score of the terminal conditions section under an n-gram model trained on the real game terminal sections?', FLOAT_FEATURE),
    FeatureDescription('ast_ngram_scoring_n_5_score', 'What is the mean 5-gram model score of the scoring section under an n-gram model trained on the real game scoring sections?', FLOAT_FEATURE),

    FeatureDescription('predicate_found_in_data_prop', 'What proportion of predicates are satisfied at least once in our human play trace data?', PROPORTION_FEATURE),
    FeatureDescription('predicate_found_in_data_small_logicals_prop', 'What proportion of logical expressions over predicates (with four or fewer children, limited for computational reasons) are satisfied at least once in our human play trace data?', PROPORTION_FEATURE),

    FeatureDescription('section_doesnt_exist_setup', 'Does a game not have an (optional) setup section? (to allow counteracting feature values for the setup for games that do not have a setup component)', BINARY_FEATURE),
    FeatureDescription('section_doesnt_exist_terminal', 'Does a game not have an (optional) terminal conditions section? (to allow counteracting feature values for the terminal conditions for games that do not have a terminal conditions component)', BINARY_FEATURE),

    FeatureDescription('variables_used_all', 'Are all variables defined used at least once?', BINARY_FEATURE),
    FeatureDescription('variables_used_prop', 'What proportion of variables defined are used at least once?', PROPORTION_FEATURE),
    FeatureDescription('preferences_used_all', 'Are all preferences defined referenced at least once in terminal or scoring expressions?', BINARY_FEATURE),
    FeatureDescription('preferences_used_prop', 'What proportion of preferences defined are referenced at least once in terminal or scoring expressions?', PROPORTION_FEATURE),
    FeatureDescription('num_preferences_defined', 'How many preferences are defined? (1-7)', DISCRETIZED_FEATURE),
    FeatureDescription('setup_objects_used', 'What proportion of objects referenced in the setup are also referenced in the gameplay preferences?', PROPORTION_FEATURE),
    FeatureDescription('any_setup_objects_used', 'Are any objects referenced in the setup also referenced in the gameplay preferences?', BINARY_FEATURE),
    FeatureDescription('repeated_variables_found', 'Are there any cases where the same variable is used twice in the same predicate?', BINARY_FEATURE),
    FeatureDescription('repeated_variable_type_in_either', 'Are there any cases where the same variable types is used twice in an \\texttt{either} quantification?', BINARY_FEATURE),
    FeatureDescription('redundant_expression_found', 'Are there any cases where a logical expression over predicates is redundant (can be trivially simplified)?', BINARY_FEATURE),
    FeatureDescription('redundant_scoring_terminal_expression_found', 'Are there any cases where a scoring terminal expression is redundant (can be trivially simplified)?', BINARY_FEATURE),
    FeatureDescription('unnecessary_expression_found', 'Are there any cases where a logical expression over predicates is unnecessary (contradicts itself, or is trivially true)?', BINARY_FEATURE),
    FeatureDescription('adjacent_same_modal_found', 'Are there any cases where the same modal is used twice in a row?', BINARY_FEATURE),
    FeatureDescription('identical_consecutive_seq_func_predicates_found', 'Are there any cases where the same exact predicates (and their arguments) are applied in consecutive modals (making them redundant)?', BINARY_FEATURE),
    FeatureDescription('disjoint_preferences_found', 'Are there any preferences which quantify over disjoint sets of objects?', BINARY_FEATURE),
    FeatureDescription('disjoint_preferences_prop', 'What proportion of preferences quantify over disjoint sets of objects?', PROPORTION_FEATURE),
    FeatureDescription('disjoint_preferences_scoring_terminal_types', 'Do the preferences referenced in the scoring and terminal section quantify over disjoint sets of object types?', PROPORTION_FEATURE),
    FeatureDescription('disjoint_preferences_scoring_terminal_predicates', 'Do the preferences referenced in the scoring and terminal section use disjoint sets of predicates?', PROPORTION_FEATURE),
    FeatureDescription('disjoint_preferences_same_predicates_only', 'Do any preferences make use solely of the \\texttt{same_color},  \\texttt{same_object}, and \\texttt{same_type} predicates?', BINARY_FEATURE),
    FeatureDescription('disjoint_seq_funcs_found', 'Are there any cases where modals in a preference refer to disjoint sets of variables or objects?', BINARY_FEATURE),
    FeatureDescription('disjoint_modal_predicates_found', 'Are there any cases where modals in a preference refer to disjoint sets of predicates?', BINARY_FEATURE),
    FeatureDescription('disjoint_modal_predicates_prop', 'What proportion of modals in a preference refer to disjoint sets of predicates?', PROPORTION_FEATURE),
    FeatureDescription('predicate_without_variables_or_agent', 'Are there any predicates that do not reference any variables or the agent?', BINARY_FEATURE),
    FeatureDescription('two_number_operation_found', 'Are there any cases where an arithmetic operation is applied to two numbers? (e.g. \\texttt{(+ 5 5)} instead of simplifying it)', BINARY_FEATURE),
    FeatureDescription('section_without_pref_or_total_count_terminal', 'Does the terminal section in this game fail to reference any preferences, or the \\texttt{(total-time)} or \\texttt{(total-score)} tokens?', BINARY_FEATURE),
    FeatureDescription('section_without_pref_or_total_count_scoring', 'Does the scoring section in this game fail to reference any preferences, or the \\texttt{(total-time)} or \\texttt{(total-score)} tokens?', BINARY_FEATURE),

    FeatureDescription('pref_forall_used_correct', 'For the \\texttt{forall} over preferences syntax, if it is used, is it used correctly in this game?', BINARY_FEATURE),
    FeatureDescription('pref_forall_used_incorrect', 'For the \\texttt{forall} over preferences syntax, if it is used, is it used incorrectly in this game? (to allow learning differential values between correct and incorrect usage)', BINARY_FEATURE),

    FeatureDescription('pref_forall_external_forall_used_correct', 'If the \\texttt{external-forall-maximize} or \\texttt{external-forall-minimize} syntax is used, is it used correctly in this game?', BINARY_FEATURE),
    FeatureDescription('pref_forall_external_forall_used_incorrect', 'If the \\texttt{external-forall-maximize} or \\texttt{external-forall-minimize} syntax is used, is it used incorrectly in this game?', BINARY_FEATURE),

    FeatureDescription('pref_forall_external_forall_used_correct', 'If the \\texttt{count-once-per-external-objects} count operator is used, is it used correctly in this game?', BINARY_FEATURE),
    FeatureDescription('pref_forall_external_forall_used_incorrect', 'If the \\texttt{count-once-per-external-objects} count operator is used, is it used incorrectly in this game?', BINARY_FEATURE),

    FeatureDescription('pref_forall_pref_forall_correct_arity_correct', 'If optional object names and types are provided to a count operation, are they provided with an arity consistent with the \\texttt{forall} variable quantifications?', BINARY_FEATURE),
    FeatureDescription('pref_forall_pref_forall_correct_arity_incorrect', 'If optional object names and types are provided to a count operation, are they provided with an arity inconsistent with the \\texttt{forall} variable quantifications?', BINARY_FEATURE),

    FeatureDescription('pref_forall_pref_forall_correct_types_correct', 'If optional object names and types are provided to a count operation, are they provided with types consistent with the \\texttt{forall} variable quantifications?', BINARY_FEATURE),
    FeatureDescription('pref_forall_pref_forall_correct_types_incorrect', 'If optional object names and types are provided to a count operation, are they provided with types inconsistent with the \\texttt{forall} variable quantifications?', BINARY_FEATURE),
]
