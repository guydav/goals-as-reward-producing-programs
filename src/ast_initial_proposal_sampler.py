import numpy as np
from scipy.stats import expon
import tatsu
import tatsu.ast
import tatsu.infos
from tqdm import trange

import ast_printer
import ast_parser
from ast_counter_sampler import *
from fitness_ngram_models import *
import fitness_energy_utils as utils
from latest_model_paths import LATEST_AST_N_GRAM_MODEL_PATH
from ast_context_fixer import ASTContextFixer


# TODO: get an argparse going


DEFAULT_NGRAM_MODEL_KWARGS = dict(
    k=7, stupid_backoff=True,
    log=True, filter_padding_top_k=True,
    top_k_min_n=5, top_k_max_n=7,
    score_all=True, k_for_sections=7,
    tokenize_entire_ast=True,
)


NGRAM_MODEL_KEY_BY_SECTION = {
    ast_parser.SETUP: 'setup_n_5_score',
    ast_parser.PREFERENCES: 'constraints_n_7_score',
    ast_parser.TERMINAL: 'terminal_n_5_score',
    ast_parser.SCORING: 'scoring_n_5_score',
}


DEFAULT_N_PREFERENCE_WEIGHTS = [0.4, 0.3, 0.3]  # [0.5, 0.35, 0.15]
DEFAULT_P_SETUP = 0.5
DEFAULT_P_TERMINAL = 0.5

# DEFAULT_MAX_DEPTH_BY_SAMPLE_SECTION = {
#     ast_parser.SETUP: 16,
#     ast_parser.PREFERENCES: 24,
#     ast_parser.TERMINAL: 10,
#     ast_parser.SCORING: 16,
# }
DEFAULT_MAX_DEPTH_BY_SAMPLE_SECTION = {
    ast_parser.SETUP: 24,
    ast_parser.PREFERENCES: 32,
    ast_parser.TERMINAL: 18,
    ast_parser.SCORING: 24,
}

# DEFAULT_SCORE_THRESHOLD_BY_SAMPLE_SECTION = {
#     ast_parser.SETUP: -3,
#     ast_parser.PREFERENCES: -4.5,
#     ast_parser.TERMINAL: -2,
#     ast_parser.SCORING: -2.75,
# }
DEFAULT_SCORE_THRESHOLD_BY_SAMPLE_SECTION = {
    ast_parser.SETUP: -4.5,  # -4,
    ast_parser.PREFERENCES: -6,  # -5.5,
    ast_parser.TERMINAL: -3.5,  # -3,
    ast_parser.SCORING: -4.5,  # -3.75,
}


class SectionBySectionNGramScoreSampler:
    context_fixer: ASTContextFixer
    max_depth_by_sample_section: typing.Dict[str, int]
    ngram_model: ASTNGramTrieModel
    ngram_model_key_by_section: typing.Dict[str, str]
    ngram_model_kwargs: typing.Dict[str, typing.Any]
    n_preferences_weights: np.ndarray
    p_setup: float
    p_terminal: float
    random_seed: int
    rng: np.random.Generator
    sample_id: int
    sampler: ASTSampler
    score_threshold_by_sample_section: typing.Dict[str, float]

    def __init__(self, sampler: ASTSampler, ngram_model: ASTNGramTrieModel, context_fixer: ASTContextFixer,
                 n_preferences_weights: typing.List[float] = DEFAULT_N_PREFERENCE_WEIGHTS,
                 p_setup: float = DEFAULT_P_SETUP, p_terminal: float = DEFAULT_P_TERMINAL,
                 max_depth_by_sample_section: typing.Dict[str, int] = DEFAULT_MAX_DEPTH_BY_SAMPLE_SECTION,
                 score_threshold_by_sample_section: typing.Dict[str, float] = DEFAULT_SCORE_THRESHOLD_BY_SAMPLE_SECTION,
                 ngram_model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 ngram_model_key_by_section: typing.Dict[str, str] = NGRAM_MODEL_KEY_BY_SECTION,
                 rng: typing.Optional[np.random.Generator] = None,
                 random_seed: int = DEFAULT_RANDOM_SEED, verbose: int = 0
                 ):
        self.sampler = sampler
        self.ngram_model = ngram_model
        self.context_fixer = context_fixer

        self.n_preferences_weights = np.array(n_preferences_weights)
        self.n_preferences_weights /= np.sum(self.n_preferences_weights)
        self.p_setup = p_setup
        self.p_terminal = p_terminal

        self.max_depth_by_sample_section = max_depth_by_sample_section
        self.score_threshold_by_sample_section = score_threshold_by_sample_section

        if ngram_model_kwargs is None:
            ngram_model_kwargs = DEFAULT_NGRAM_MODEL_KWARGS
        else:
            temp_kwargs = DEFAULT_NGRAM_MODEL_KWARGS.copy()
            temp_kwargs.update(ngram_model_kwargs)
            ngram_model_kwargs = temp_kwargs

        self.ngram_model_kwargs = ngram_model_kwargs
        self.ngram_model_key_by_section = ngram_model_key_by_section

        if rng is None:
            rng = np.random.default_rng(random_seed)
        self.rng = rng
        self.random_seed = random_seed
        self.verbose = verbose
        self.sample_id = 0

    def _sample_rule(self, rule: str, section: str, global_context: typing.Dict[str, typing.Any]) -> tatsu.ast.AST:
        self.sampler.max_sample_depth = self.max_depth_by_sample_section[section]
        ngram_ast_parser_kwargs = {ast_parser.SECTION_CONTEXT_KEY: section}

        sample = None
        sampled = False
        sample_global_context = {}

        while not sampled:
            try:
                sample_global_context = simplified_context_deepcopy(global_context)
                sample = self.sampler.sample(rule, global_context=sample_global_context)[0]
                ngram_score_dict = self.ngram_model.score(
                    sample, **self.ngram_model_kwargs,  # type: ignore
                    ngram_ast_parser_kwargs=ngram_ast_parser_kwargs
                )
                ngram_score = typing.cast(float, ngram_score_dict[self.ngram_model_key_by_section[section]])
                if ngram_score is None:
                    raise ValueError(f'No ngram score for section {section} with key {self.ngram_model_key_by_section[section]}')

                if ngram_score > self.score_threshold_by_sample_section[section]:
                    if self.verbose: print(f'Sampled sample with ngram score {ngram_score} for section {section}')
                    sampled = True
                else:
                    exp_pdf = expon.pdf(-ngram_score, loc=-self.score_threshold_by_sample_section[section])
                    if self.rng.uniform() < exp_pdf:
                        if self.verbose:
                            print(f'Sampled sample with ngram score {ngram_score} for section {section} with exp_pdf {exp_pdf}')
                        sampled = True

                    elif self.verbose:
                        print(f'Skipping sample with ngram score {ngram_score} for section {section}')

            except SamplingException as e:
                if self.verbose:
                    print(f'Failed to sample {rule} with global context {global_context}: {e}')
                continue

        global_context.update(sample_global_context)
        return sample  # type: ignore

    def sample(self, global_context: typing.Optional[typing.Dict[str, typing.Any]] = None):
        sampled = False
        candidate = None

        current_sampler_max_depth = self.sampler.max_sample_depth

        if global_context is None:
            global_context = {}

        if 'sample_id' not in global_context:
            global_context['sample_id'] = self.sample_id

        while not sampled:
            try:
                setup = None
                if self.rng.uniform() < self.p_setup:
                    setup = self._sample_rule('setup', ast_parser.SETUP, global_context)
                    setup = ('(:setup', setup, ')')

                n_preferences = self.rng.choice(np.arange(1, len(self.n_preferences_weights) + 1), p=self.n_preferences_weights)
                preference_list = [self._sample_rule('pref_def', ast_parser.PREFERENCES, global_context) for _ in range(n_preferences) ]

                preferences_dict = dict(preferences=preference_list)
                preferences_dict['parseinfo'] = tatsu.infos.ParseInfo(  # type: ignore
                    None, 'preferences', self.sampler.sample_parseinfo_index, self.sampler.sample_parseinfo_index,
                    self.sampler.sample_parseinfo_index, self.sampler.sample_parseinfo_index)
                self.sampler.sample_parseinfo_index += 1
                preferences = ('(:constraints', tatsu.ast.AST(preferences_dict), ')')

                terminal = None
                if self.rng.uniform() < self.p_terminal:
                    terminal = self._sample_rule('terminal', ast_parser.TERMINAL, global_context)
                    terminal = ('(:terminal', terminal, ')')

                scoring = self._sample_rule('scoring_expr', ast_parser.SCORING, global_context)
                scoring = ('(:scoring', scoring, ')')

                game_def = self.sampler.sample('game_def', global_context=global_context)[0]
                domain_def = self.sampler.sample('domain_def', global_context=global_context)[0]

                game_sections = ['(define', game_def, domain_def]
                if setup is not None: game_sections.append(setup)
                game_sections.append(preferences)
                if terminal is not None: game_sections.append(terminal)
                game_sections.append(scoring)
                game_sections.append(')')

                candidate = tuple(game_sections)
                self.context_fixer.fix_contexts(candidate)  # type: ignore
                sampled = True

            except SamplingException as e:
                if self.verbose:
                    print(f'Failed to sample game with global context {global_context}: {e}')
                continue

        self.sample_id += 1
        self.sampler.max_sample_depth = current_sampler_max_depth

        return candidate


if __name__ == '__main__':
    grammar = open('./dsl/dsl.ebnf').read()
    grammar_parser = tatsu.compile(grammar)

    DEFUALT_RANDOM_SEED = 33
    DEFAULT_ARGS = argparse.Namespace(
        grammar_file=os.path.join('.', DEFAULT_GRAMMAR_FILE),
        parse_counter=True,
        counter_output_path=os.path.join('.', DEFAULT_COUNTER_OUTPUT_PATH),
        random_seed=DEFUALT_RANDOM_SEED,
        test_files=[os.path.join('.', f) for f in DEFAULT_TEST_FILES],
        dont_tqdm=True,
        relative_path='.',
    )

    counter = parse_or_load_counter(DEFAULT_ARGS, grammar_parser=grammar_parser)
    sampler = ASTSampler(grammar_parser, counter, seed=DEFAULT_ARGS.random_seed)  # type: ignore
    context_fixer = ASTContextFixer(sampler, sampler.rng, True)

    with open(LATEST_AST_N_GRAM_MODEL_PATH, 'rb') as f:
        ngram_model = pickle.load(f)

    section_sampler = SectionBySectionNGramScoreSampler(sampler, ngram_model, context_fixer, verbose=0)

    N_SAMPLES = 100

    from ast_mcmc_regrowth import *

    mcmc = utils.load_data('2023_04_11', 'samples', f'mcmc_full_features_0_5', relative_path='.')

    section_sampler_energy_scores = []
    map_sampler_energy_scores = []

    for _ in trange(N_SAMPLES):
        ast = section_sampler.sample()
        section_sampler_energy_scores.append(mcmc._score_proposal(ast)[1])
        # print(ast_printer.ast_to_string(ast, '\n'))  # type: ignore
        # print()

    sampler.max_sample_depth = 30

    for _ in trange(N_SAMPLES):
        ast = sampler.sample_until_success()
        map_sampler_energy_scores.append(mcmc._score_proposal(ast)[1])
        # print(ast_printer.ast_to_string(ast, '\n'))  # type: ignore
        # print()

    print(f'Average section sampler energy score: {np.mean(section_sampler_energy_scores)} +- {np.std(section_sampler_energy_scores)}')
    print(f'Average MAP sampler energy score: {np.mean(map_sampler_energy_scores)} +- {np.std(map_sampler_energy_scores)}')
