from dataclasses import dataclass
import os
import typing

_FILE_DIR = os.path.dirname(__file__)


LATEST_REAL_GAMES_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../dsl/interactive-beta.pddl'))

LATEST_FITNESS_FEATURES = os.path.abspath(os.path.join(_FILE_DIR,  '../data/fitness_features_1024_regrowths.csv.gz'))
LATEST_AST_N_GRAM_MODEL_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/ast_5_ngram_model_2023_11_22.pkl'))
# LATEST_AST_N_GRAM_MODEL_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/ast_7_ngram_model_2023_11_13.pkl'))
LATEST_FITNESS_FEATURIZER_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/fitness_featurizer_2023_12_22.pkl.gz'))
LATEST_FITNESS_FUNCTION_DATE_ID = 'cv_fitness_model_in_data_prop_L2_categories_minimal_counting_grammar_use_forall_seed_42_2023_12_22'

# LATEST_FITNESS_FEATURES_SPECIFIC_OBJECTS_NGRAM = os.path.abspath(os.path.join(_FILE_DIR,  '../data/fitness_features_1024_regrowths_specific_objects_ngram.csv.gz'))
# LATEST_SPECIFIC_OBJECTS_AST_N_GRAM_MODEL_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/ast_specific_objects_5_ngram_model_2023_06_27.pkl'))
# LATEST_SPECIFIC_OBJECTS_FITNESS_FEATURIZER_PATH = os.path.abspath(os.path.join(_FILE_DIR, '../models/fitness_featurizer_specific_objects_ngram_2023_07_21.pkl.gz'))
# LATEST_SPECIFIC_OBJECTS_FITNESS_FUNCTION_DATE_ID = 'full_features_specific_objects_ngram_2023_07_21'


@dataclass
class MapElitesModelSpec:
    save_path: str
    name: str
    trace_filter_suffix: typing.Optional[str] = None

    def get_different_gen(self, gen: typing.Union[str, int], date: typing.Optional[str] = None):
        gen_index = self.save_path.find('gen')
        if gen_index == -1:
            gen_index = self.save_path.find('final')

        if gen_index == -1:
            raise ValueError(f'Could not find gen in {self.save_path}')

        date_index = self.save_path.find('2023')
        if date_index == -1:
            date_index = self.save_path.find('2024')

        if date_index == -1:
            raise ValueError(f'Could not find date in {self.save_path}')

        if date is None:
            date = self.save_path[date_index:]

        if isinstance(gen, int) or gen.isdigit():
            gen = f'gen_{gen}'

        return f'{self.save_path[:gen_index]}_{gen}_{date}'

    def load(self, gen: typing.Optional[typing.Union[str, int]] = None, date: typing.Optional[str] = None,
             date_and_id: str = '', folder: str = 'samples', relative_path: str = '..'):
        load_path = self.save_path if gen is None else self.get_different_gen(gen, date)
        from fitness_energy_utils import load_data
        return load_data(date_and_id, folder, load_path, relative_path=relative_path)

    def load_trace_filter_data(self):
        if self.trace_filter_suffix is None:
            return None

        from fitness_energy_utils import load_data
        return load_data('', 'samples', f'trace_filter_results_{self.save_path}_{self.trace_filter_suffix}', relative_path='..')


MAP_ELITES_MODELS = {
    # 'previous_object_and_predicates_with_expected_values_bc': MapElitesModelSpec(
    #     'map_elites_minimal_counting_grammar_use_forall_L2_latest_setup_expected_values_uniform_seed_42_gen_4096_2023_10_25',
    #     'Using New "Goodness" BC | Uniform archive sampling | L2 regularization'
    # ),

    # 'objects_and_predicates': MapElitesModelSpec(
    #     'map_elites_minimal_counting_grammar_use_forall_L2_latest_setup_uniform_seed_42_gen_4096_2023_11_14',
    #     'No "Goodness" BC | Uniform archive sampling | L2 regularization'
    # ),
    'object_and_predicates_with_expected_values_bc': MapElitesModelSpec(
        'map_elites_minimal_counting_grammar_use_forall_L2_latest_setup_expected_values_uniform_seed_42_final_2023_11_29',
        'Object and Predicate Group BCs | Using "Goodness" BC'
    ),

    'object_and_predicates_with_at_end_expected_values_bc': MapElitesModelSpec(
        'map_elites_minimal_counting_grammar_use_forall_L2_latest_at_end_no_game_object_expected_values_uniform_seed_42_final_2023_11_29',
        'Object and Predicate Group BCs | No game_object, with at_end | Using "Goodness" BC'
    ),

    'predicate_and_object_groups_setup_at_end_pref_count_expected_values_bc': MapElitesModelSpec(
        'map_elites_minimal_counting_grammar_use_forall_L2_predicate_and_object_groups_setup_at_end_pref_count_expected_values_uniform_seed_42_gen_8192_final_2024_01_02',
        'Latest Object and Predicate Group BCs | With at_end, setup, num preferences | Using "Goodness" BC',
        '2024_01_10'
    ),

    # 'num_exemplar_preferences_by_bcs': MapElitesModelSpec(
    #     'map_elites_minimal_counting_grammar_use_forall_L2_exemplar_preferences_bc_num_prefs_setup_uniform_seed_42_gen_4096_2023_11_17',
        # 'Exemplar Preferenecs BCs | Binary BCs, no max preference count | No "Goodness" BC',
    # ),
    'num_exemplar_preferences_by_bcs_with_expected_values': MapElitesModelSpec(
        'map_elites_minimal_counting_grammar_use_forall_L2_exemplar_preferences_bc_num_prefs_expected_values_uniform_seed_42_gen_8192_final_2023_12_24',
        'Exemplar Preferenecs BCs | Binary BCs, no max preference count | Using "Goodness" BC',
        '2023_12_24',
    ),
    # 'max_exemplar_preferences_by_bcs': MapElitesModelSpec(
    #     'map_elites_minimal_counting_grammar_use_forall_L2_exemplar_preferences_bc_max_prefs_setup_uniform_seed_42_gen_4096_2023_11_21',
    #     'Exemplar Preferenecs BCs | Count BCs, with max preference count | No "Goodness" BC',
    # ),
    'max_exemplar_preferences_by_bcs_with_expected_values': MapElitesModelSpec(
        # 'map_elites_minimal_counting_grammar_use_forall_L2_exemplar_preferences_bc_max_prefs_expected_values_uniform_seed_42_gen_4096_final_2023_11_29',
        'map_elites_minimal_counting_grammar_use_forall_L2_exemplar_preferences_bc_max_prefs_expected_values_uniform_seed_42_gen_8192_final_2023_12_23',
        'Exemplar Preferenecs BCs | Count BCs, with max preference count | Using "Goodness" BC',
        '2023_12_24',
    ),

    'ablation_max_exemplar_preferences_by_bcs_with_expected_values_no_custom_ops': MapElitesModelSpec(
        'map_elites_minimal_counting_grammar_use_forall_L2_no_cognitive_ops_exemplar_preferences_bc_max_prefs_expected_values_uniform_seed_42_gen_8192_final_2024_01_18',
        'Exemplar Preferenecs BCs | Count BCs, with max preference count | Using "Goodness" BC | No custom ops',
        '2024_03_20',
    ),

    'ablation_max_exemplar_preferences_by_bcs_with_expected_values_no_custom_ops_no_crossover': MapElitesModelSpec(
        'map_elites_minimal_counting_grammar_use_forall_L2_no_cognitive_ops_no_crossover_exemplar_preferences_bc_max_prefs_expected_values_uniform_seed_42_gen_8192_final_2024_01_17',
        'Exemplar Preferenecs BCs | Count BCs, with max preference count | Using "Goodness" BC | No custom ops and no crossover',
        '2024_03_20',
    ),

    'ablation_max_exemplar_preferences_by_bcs_with_expected_values_no_play_trace_features': MapElitesModelSpec(
        'map_elites_minimal_counting_grammar_use_forall_L2_no_play_trace_database_features_exemplar_preferences_bc_max_prefs_expected_values_uniform_seed_42_gen_8192_final_2024_03_26',
        'Exemplar Preferenecs BCs | Count BCs, with max preference count | Using "Goodness" BC | No play trace features',
        '2024_03_26',
    ),

    'ablation_max_exemplar_preferences_by_bcs_with_expected_values_no_coherence_features': MapElitesModelSpec(
        'map_elites_minimal_counting_grammar_use_forall_L2_no_game_element_disjointness_features_exemplar_preferences_bc_max_prefs_expected_values_uniform_seed_42_gen_8192_final_2024_03_26',
        'Exemplar Preferenecs BCs | Count BCs, with max preference count | Using "Goodness" BC | No coherence features',
        '2024_04_13',
    ),

    'bcs_ablation_predicate_and_object_groups_setup_at_end_pref_count_expected_values': MapElitesModelSpec(
        'map_elites_minimal_counting_grammar_use_forall_L2_bcs_ablation_predicate_and_object_groups_setup_at_end_pref_count_expected_values_uniform_seed_42_gen_8192_final_2024_03_28',
        'BC Ablations | Predicate and object groups with at_end, setup, pref count | Using "Goodness" BC ',
        '2024_04_03',
    ),

    'bcs_ablation_latest_at_end_no_game_object_expected_values': MapElitesModelSpec(
        'map_elites_minimal_counting_grammar_use_forall_L2_bcs_ablation_latest_at_end_no_game_object_expected_values_uniform_seed_42_gen_8192_final_2024_03_28',
        'BC Ablations | Predicate and object groups with no game object | Using "Goodness" BC ',
        '2024_04_02',
    ),

    'prior_sampling_only': MapElitesModelSpec(
        'map_elites_prior_sampling_only_exemplar_preferences_bc_max_prefs_expected_values_uniform_seed_42_gen_8192_final_2024_08_15',
        'Prior Sampling Only | Exemplar Preferenecs BCs | Count BCs, with max preference count | Using "Goodness" BC',
        '2024_08_15',
    ),
}
