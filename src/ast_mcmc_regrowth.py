import argparse
import enum
import gzip
import logging
from multiprocessing import pool as mpp
import os
import pickle
import typing
import sys

import numpy as np
from tqdm import tqdm, trange, notebook
import tatsu
import tatsu.ast
import tatsu.grammars
import torch

from ast_counter_sampler import *
from ast_initial_proposal_sampler import SectionBySectionNGramScoreSampler, NGRAM_MODEL_KEY_BY_SECTION
from ast_initial_proposal_sampler import *
from fitness_ngram_models import *
from ast_parser import ASTSamplePostprocessor
import ast_printer
from ast_utils import cached_load_and_parse_games_from_file
from fitness_features import *
from fitness_energy_utils import NON_FEATURE_COLUMNS, evaluate_single_game_energy_contributions, evaluate_comparison_energy_contributions, load_model_and_feature_columns, DEFAULT_SAVE_MODEL_NAME, save_data
from latest_model_paths import LATEST_AST_N_GRAM_MODEL_PATH, LATEST_FITNESS_FEATURIZER_PATH, LATEST_FITNESS_FUNCTION_DATE_ID

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


parser = argparse.ArgumentParser(description='MCMC Regrowth Sampler')
parser.add_argument('--grammar-file', type=str, default=DEFAULT_GRAMMAR_FILE)
parser.add_argument('--parse-counter', action='store_true')
parser.add_argument('--counter-output-path', type=str, default=DEFAULT_COUNTER_OUTPUT_PATH)

DEFAULT_FITNESS_FUNCTION_DATE_ID = LATEST_FITNESS_FUNCTION_DATE_ID
parser.add_argument('--fitness-function-date-id', type=str, default=DEFAULT_FITNESS_FUNCTION_DATE_ID)
DEFAULT_FITNESS_FEATURIZER_PATH = LATEST_FITNESS_FEATURIZER_PATH
parser.add_argument('--fitness-featurizer-path', type=str, default=DEFAULT_FITNESS_FEATURIZER_PATH)
parser.add_argument('--fitness-function-model-name', type=str, default=DEFAULT_SAVE_MODEL_NAME)

DEFAULT_PLATEAU_PATIENCE_STEPS = 1000
parser.add_argument('--plateau-patience-steps', type=int, default=DEFAULT_PLATEAU_PATIENCE_STEPS)
DEFAULT_MAX_STEPS = 20000
parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS)
DEFAULT_N_SAMPLES_PER_STEP = 1
parser.add_argument('--n-samples-per-step', type=int, default=DEFAULT_N_SAMPLES_PER_STEP)
parser.add_argument('--non-greedy', action='store_true')
DEFAULT_ACCEPTANCE_TEMPERATURE = 1.0
parser.add_argument('--acceptance-temperature', type=float, default=DEFAULT_ACCEPTANCE_TEMPERATURE)
DEFAULT_RELATIVE_PATH = '.'
parser.add_argument('--relative-path', type=str, default=DEFAULT_RELATIVE_PATH)
DEFAULT_NGRAM_MODEL_PATH = LATEST_AST_N_GRAM_MODEL_PATH
parser.add_argument('--ngram-model-path', type=str, default=DEFAULT_NGRAM_MODEL_PATH)
DEFUALT_RANDOM_SEED = 33
parser.add_argument('--random-seed', type=int, default=DEFUALT_RANDOM_SEED)

parser.add_argument('--initial-proposal-type', type=int, default=0)
parser.add_argument('--start-from-real-games', action='store_true')
parser.add_argument('--real-games-path', type=str, default='./dsl/interactive-beta.pddl')
parser.add_argument('--n-samples', type=int, default=10)
parser.add_argument('--real-game-start-index', type=int, default=0)
parser.add_argument('--real-game-end-index', type=int, default=10)
parser.add_argument('--sample-parallel', action='store_true')
parser.add_argument('--parallel-n-workers', type=int, default=8)
parser.add_argument('--parallel-chunksize', type=int, default=1)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--should-tqdm', action='store_true')
parser.add_argument('--postprocess', action='store_true')

DEFAULT_OUTPUT_NAME = 'mcmc'
parser.add_argument('--output-name', type=str, default=DEFAULT_OUTPUT_NAME)
parser.add_argument('--output-folder', type=str, default='./samples')


class InitialProposalSamplerType(enum.Enum):
    MAP = 0
    SECTION_SAMPLER = 1


def create_initial_proposal_sampler(initial_proposal_type: InitialProposalSamplerType,
                                    sampler: ASTSampler, context_fixer: typing.Callable,
                                    ngram_model_path: typing.Optional[str] = None,
                                    section_sampler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None):
    if initial_proposal_type == InitialProposalSamplerType.MAP:
        return sampler
    elif initial_proposal_type == InitialProposalSamplerType.SECTION_SAMPLER:
        if section_sampler_kwargs is None:
            section_sampler_kwargs = {}

        if ngram_model_path is None:
            raise ValueError('ngram_model_path must be specified for section sampler')

        ngram_model_index = ngram_model_path.find('ngram_model')
        model_n = int(ngram_model_path[ngram_model_index - 2: ngram_model_index - 1])
        if model_n != 7:
            if 'ngram_model_key_by_section' not in section_sampler_kwargs:
                section_sampler_kwargs['ngram_model_key_by_section'] = {k: f'{k[2:]}_n_{model_n}_score' for k in ast_parser.SECTION_KEYS}

            if 'ngram_model_kwargs' not in section_sampler_kwargs:
                section_sampler_kwargs['ngram_model_kwargs'] = {}

            if 'k' not in section_sampler_kwargs['ngram_model_kwargs']:
                section_sampler_kwargs['ngram_model_kwargs']['k'] = model_n

            if 'top_k_min_n' not in section_sampler_kwargs['ngram_model_kwargs']:
                section_sampler_kwargs['ngram_model_kwargs']['top_k_min_n'] = model_n

            if 'top_k_max_n' not in section_sampler_kwargs['ngram_model_kwargs']:
                section_sampler_kwargs['ngram_model_kwargs']['top_k_max_n'] = model_n

            if 'k_for_sections' not in section_sampler_kwargs['ngram_model_kwargs']:
                section_sampler_kwargs['ngram_model_kwargs']['k_for_sections'] = model_n

        with open(ngram_model_path, 'rb') as f:
            ngram_model = pickle.load(f)

        return SectionBySectionNGramScoreSampler(sampler, ngram_model, context_fixer, **section_sampler_kwargs)


def _load_pickle_gzip(path: str):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


class MCMCRegrowthSampler:
    acceptance_temperature: float
    counter: ASTRuleValueCounter
    feature_names: typing.List[str]
    fitness_featurizer: ASTFitnessFeaturizer
    fitness_featurizer_path: str
    fitness_function: typing.Callable
    fitness_function_date_id: str
    fitness_function_model_name: str
    grammar: str
    grammar_parser: tatsu.grammars.Grammar  # type: ignore
    greedy_acceptance: bool
    initial_proposal_sampler: typing.Union[ASTSampler, SectionBySectionNGramScoreSampler]
    initial_proposal_type: InitialProposalSamplerType
    max_steps: int
    n_samples_per_step: int
    plateau_patience_steps: int
    postprocessor: ASTSamplePostprocessor
    regrowth_sampler: RegrowthSampler
    sampler: ASTSampler
    samples: typing.List[tuple]

    def __init__(self,
        args: argparse.Namespace,
        fitness_function_date_id: str = DEFAULT_FITNESS_FUNCTION_DATE_ID,
        fitness_featurizer_path: str = DEFAULT_FITNESS_FEATURIZER_PATH,
        initial_proposal_type: InitialProposalSamplerType = InitialProposalSamplerType.MAP,
        max_steps: int = DEFAULT_MAX_STEPS,
        n_samples_per_step: int = DEFAULT_N_SAMPLES_PER_STEP,
        greedy_acceptance: bool = False,
        acceptance_temperature: float = DEFAULT_ACCEPTANCE_TEMPERATURE,
        plateau_patience_steps: int = DEFAULT_PLATEAU_PATIENCE_STEPS,
        fitness_function_model_name: str = DEFAULT_SAVE_MODEL_NAME,
        relative_path: str = DEFAULT_RELATIVE_PATH,
        ngram_model_path: str = DEFAULT_NGRAM_MODEL_PATH,
        section_sampler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ):
        self.args = args

        self.grammar = open(args.grammar_file).read()
        self.grammar_parser = tatsu.compile(self.grammar)  # type: ignore
        self.counter = parse_or_load_counter(args, self.grammar_parser)
        self.sampler = ASTSampler(self.grammar_parser, self.counter, seed=args.random_seed)
        self.regrowth_sampler = RegrowthSampler(self.sampler, seed=args.random_seed)
        self.random_seed = args.random_seed

        self.fitness_featurizer_path = fitness_featurizer_path
        self.fitness_featurizer = _load_pickle_gzip(fitness_featurizer_path)
        self.fitness_function_date_id = fitness_function_date_id
        self.fitness_function_model_name = fitness_function_model_name
        self.fitness_function, self.feature_names = load_model_and_feature_columns(fitness_function_date_id, name=fitness_function_model_name, relative_path=relative_path)  # type: ignore
        self.plateau_patience_steps = plateau_patience_steps
        self.max_steps = max_steps
        self.n_samples_per_step = n_samples_per_step
        self.greedy_acceptance = greedy_acceptance
        self.acceptance_temperature = acceptance_temperature
        self.initial_proposal_type = initial_proposal_type

        self.initial_proposal_sampler = create_initial_proposal_sampler(
            initial_proposal_type, sampler, ngram_model_path, section_sampler_kwargs)

        self.postprocessor = ASTSamplePostprocessor()

        self.samples = []

    def _evaluate_fitness(self, features: torch.Tensor):
        if 'wrapper' in self.fitness_function.named_steps:  # type: ignore
            self.fitness_function.named_steps['wrapper'].eval()  # type: ignore
        return self.fitness_function.transform(features).item()

    def visualize_sample_with_energy_difference(self, sample_index: int, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005):
        energy_diff_sample_indices = [i for i, s in enumerate(self.samples) if len(s) > 3 and s[2] < s[5]]
        original_sample_index = energy_diff_sample_indices[sample_index]
        print(f'Sample #{sample_index} with an energy difference is the the original sample #{original_sample_index}')
        self.visualize_sample(original_sample_index, display_overall_features=display_overall_features, display_game=display_game, min_display_threshold=min_display_threshold)

    def visualize_sample(self, sample_index: int, top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005, postprocess_sample: bool = False):
        sample_tuple = self.samples[sample_index]
        sample = sample_tuple[0]
        sample_features_tensor = self._features_to_tensor(sample_tuple[1])

        if len(self.samples[sample_index]) > 6:
            original = sample_tuple[-3]
            original_features_tensor = self._features_to_tensor(sample_tuple[-2])
            evaluate_comparison_energy_contributions(
                original_features_tensor, sample_features_tensor, ast_printer.ast_to_string(original, '\n'), ast_printer.ast_to_string(sample, '\n'),
                self.fitness_function, self.feature_names, top_k=top_k, min_display_threshold=min_display_threshold,  # type: ignore
            )

        else:
            if postprocess_sample:
                sample = self.postprocessor(sample)

            evaluate_single_game_energy_contributions(
                self.fitness_function, sample_features_tensor, ast_printer.ast_to_string(sample, '\n'), self.feature_names,  # type: ignore
                top_k=top_k, display_overall_features=display_overall_features,
                display_game=display_game, min_display_threshold=min_display_threshold,
                )

    def _generate_sample_parallel(self, sample_index: int, initial_proposal: typing.Optional[typing.Union[tatsu.ast.AST, tuple]] = None,
                                  verbose: int = 0, postprocess: typing.Optional[bool] = None):
        rng = np.random.default_rng(self.random_seed + sample_index)
        return (sample_index, self.sample(verbose=verbose, initial_proposal=initial_proposal, postprocess=postprocess,
                                          return_sample=True, sample_index=sample_index, rng=rng))


    def multiple_samples_parallel(self, n_samples_per_initial_proposal: int, verbose: int = 0, should_tqdm: bool = False,
                                  initial_proposals: typing.Optional[typing.List[typing.Union[None, tatsu.ast.AST, tuple]]] = None,
                                  postprocess: typing.Optional[bool] = None, n_workers: int = 8,
                                  chunksize: int = 1, notebook_tqdm: bool = False):

        logger.debug(f'Launching multiprocessing pool with {n_workers} workers...')
        with mpp.Pool(n_workers) as pool:
            if initial_proposals is None:
                initial_proposals = [None] * n_samples_per_initial_proposal  # type: ignore
            else:
                initial_proposals = [proposal for proposal in initial_proposals for _ in range(n_samples_per_initial_proposal)]

            sample_param_iter = zip(range(len(initial_proposals)), initial_proposals, [verbose] * len(initial_proposals), [postprocess] * len(initial_proposals))  # type: ignore
            samples = {}

            pool_iter = pool.istarmap(self._generate_sample_parallel, sample_param_iter, chunksize=chunksize)  # type: ignore

            if should_tqdm:
                if notebook_tqdm:
                    pool_iter = notebook.tqdm(pool_iter, total=len(initial_proposals))  # type: ignore
                else:
                    pool_iter = tqdm(pool_iter, total=len(initial_proposals))  # type: ignore

            for index, sample in pool_iter:
                samples[index] = sample

        for index in sorted(samples.keys()):
            self.samples.append(samples[index])

    def multiple_samples(self, n_samples: int, verbose: int = 0, should_tqdm: bool = False,
                         initial_proposal: typing.Optional[typing.Union[tatsu.ast.AST, tuple]] = None, postprocess: typing.Optional[bool] = None,
                         notebook_tqdm: bool = False):
        sample_iter = (notebook.trange(n_samples) if notebook_tqdm else trange(n_samples)) if should_tqdm else range(n_samples)
        for _ in sample_iter:
            self.sample(verbose=verbose, initial_proposal=initial_proposal, postprocess=postprocess)

    def sample(self, verbose: int = 0, initial_proposal: typing.Optional[typing.Union[tatsu.ast.AST, tuple]] = None,
               postprocess: typing.Optional[bool] = None, return_sample: bool = False, sample_index: typing.Optional[int] = None,
               rng: typing.Optional[np.random.Generator] = None):
        initial_proposal_provided = initial_proposal is not None
        if postprocess is None:
            postprocess = not initial_proposal_provided

        if sample_index is None:
            sample_index = len(self.samples)

        if rng is None:
            rng = np.random.default_rng()

        while initial_proposal is None:
            try:
                initial_proposal = typing.cast(tuple, self.initial_proposal_sampler.sample(global_context=dict(rng=rng, original_game_id=f'mcmc-{sample_index}')))
            except RecursionError:
                if verbose >= 2: print('Recursion error, skipping sample')
            except SamplingException:
                if verbose >= 2: print('Sampling exception, skipping sample')

        initial_proposal_features, initial_proposal_fitness = self._score_proposal(initial_proposal)  # type: ignore

        current_proposal, current_proposal_features, current_proposal_fitness = initial_proposal, initial_proposal_features, initial_proposal_fitness
        best_proposal, best_proposal_features, best_proposal_fitness = initial_proposal, initial_proposal_features, initial_proposal_fitness

        accepted_steps = []
        improved_steps = []

        best_proposal_step = 0

        for step in range(self.max_steps):
            current_proposal, current_proposal_features, current_proposal_fitness, accepted = self.mcmc_step(
                current_proposal, current_proposal_features, current_proposal_fitness, step, verbose, rng
            )

            if accepted:
                accepted_steps.append(step)
                if verbose:
                    print(f'Accepted step {step} with energy {current_proposal_fitness:.5f}', end='\r')

                if current_proposal_fitness < best_proposal_fitness:  # always when this is greedy, sometimes when not
                    best_proposal, best_proposal_features, best_proposal_fitness = current_proposal, current_proposal_features, current_proposal_fitness
                    improved_steps.append(step)
                    best_proposal_step = step
                    if verbose:
                        print(f'New best proposal at step {step} with energy {current_proposal_fitness:.5f}')


            if step - best_proposal_step > self.plateau_patience_steps:
                if verbose:
                    if initial_proposal_provided:
                        print(f'Plateaued at step {step} with energy {best_proposal_fitness:.5f} (initial proposal energy: {initial_proposal_fitness:.5f})')
                    else:
                        print(f'Plateaued at step {step} with energy {best_proposal_fitness:.5f}')
                break

        if postprocess:
            best_proposal = self.postprocessor(best_proposal)

        if initial_proposal_provided:
            sample_tuple = (best_proposal, best_proposal_features, best_proposal_fitness, best_proposal_step, tuple(accepted_steps), tuple(improved_steps), initial_proposal, initial_proposal_features, initial_proposal_fitness)
        else:
            sample_tuple = (best_proposal, best_proposal_features, best_proposal_fitness, best_proposal_step, tuple(accepted_steps), tuple(improved_steps),)

        if return_sample:
            return sample_tuple

        self.samples.append(sample_tuple)

    def _generate_step_proposal(self, step_index: int, rng: typing.Optional[np.random.Generator] = None) -> typing.Union[tatsu.ast.AST, tuple]:
        return self.regrowth_sampler.sample(step_index, update_game_id=False, rng=rng)

    def _pre_mcmc_step(self, current_proposal: typing.Union[tatsu.ast.AST, tuple]):
        if self.regrowth_sampler.source_ast != current_proposal:
            self.regrowth_sampler.set_source_ast(current_proposal)

    def mcmc_step(self,
        current_proposal: typing.Union[tatsu.ast.AST, tuple],
        current_proposal_features: typing.Dict[str, float],
        current_proposal_fitness: float,
        step_index: int,
        verbose: int = 0,
        rng: typing.Optional[np.random.Generator] = None,
        ) -> typing.Tuple[typing.Union[tatsu.ast.AST, tuple], typing.Dict[str, float], float, bool] :

        self._pre_mcmc_step(current_proposal)

        new_proposal = None
        samples = []

        while len(samples) < self.n_samples_per_step:
            try:
                new_proposal = self._generate_step_proposal(step_index, rng)
                # _test_ast_sample(ast, args, text_samples, grammar_parser)
                if ast_printer.ast_to_string(new_proposal) == ast_printer.ast_to_string(current_proposal):  # type: ignore
                    if verbose >= 2: print('Regrowth generated identical games, repeating')
                else:
                    samples.append(new_proposal)

            except RecursionError:
                if verbose >= 2: print('Recursion error, skipping sample')

            except SamplingException:
                if verbose >= 2: print('Sampling exception, skipping sample')

        samples = [(sample, *self._score_proposal(sample)) for sample in samples]
        new_proposal, new_proposal_features, new_proposal_fitness = min(samples, key=lambda x: x[2])

        if self.greedy_acceptance:
            accept = new_proposal_fitness < current_proposal_fitness
        else:
            acceptance_probability = np.exp(-self.acceptance_temperature * (new_proposal_fitness - current_proposal_fitness))
            if rng is None:
                rng = np.random.default_rng(self.random_seed + step_index)
            accept = rng.uniform() < acceptance_probability

        if accept:
            return new_proposal, new_proposal_features, new_proposal_fitness, True
        else:
            return current_proposal, current_proposal_features, current_proposal_fitness, False

    def _proposal_to_features(self, proposal: tatsu.ast.AST):
        return typing.cast(dict, self.fitness_featurizer.parse(proposal, 'mcmc', True))  # type: ignore

    def _features_to_tensor(self, features: typing.Dict[str, typing.Any]):
        return torch.tensor([features[name] for name in self.feature_names], dtype=torch.float32)  # type: ignore

    def _score_proposal(self, proposal: tatsu.ast.AST):
        proposal_features = self._proposal_to_features(proposal)
        proposal_tensor = self._features_to_tensor(proposal_features)
        proposal_fitness = self._evaluate_fitness(proposal_tensor)
        return proposal_features, proposal_fitness


# class MCMCRegrowthCrossoverSampler(MCMCRegrowthSampler):
#     def __init__(self,
#         args: argparse.Namespace,
#         crossover_type: CrossoverType,
#         crossover_population: typing.List[typing.Union[tatsu.ast.AST, tuple]],
#         p_crossover: float,
#         fitness_function_date_id: str = DEFAULT_FITNESS_FUNCTION_DATE_ID,
#         fitness_featurizer_path: str = DEFAULT_FITNESS_FEATURIZER_PATH,
#         plateau_patience_steps: int = DEFAULT_PLATEAU_PATIENCE_STEPS,
#         max_steps: int = DEFAULT_MAX_STEPS,
#         greedy_acceptance: bool = False,
#         acceptance_temperature: float = DEFAULT_ACCEPTANCE_TEMPERATURE,
#         fitness_function_model_name: str = DEFAULT_SAVE_MODEL_NAME,
#         fitness_function_relative_path: str = '.',
#         ):
#         super().__init__(args=args,
#             fitness_function_date_id=fitness_function_date_id, fitness_featurizer_path=fitness_featurizer_path,
#             plateau_patience_steps=plateau_patience_steps, max_steps=max_steps,
#             greedy_acceptance=greedy_acceptance, acceptance_temperature=acceptance_temperature,
#             fitness_function_model_name=fitness_function_model_name, relative_path=fitness_function_relative_path,
#         )
#         self.crossover_sampler = CrossoverSampler(crossover_type, crossover_population, self.sampler, args.random_seed)
#         self.p_crossover = p_crossover
#         self.step_index_sampled = -1
#         self.crossover_current_step = None

#     def _pre_mcmc_step(self, current_proposal: tatsu.ast.AST):
#         # TODO: consider in the future to optimize setting source ast only to the sampler chosen for this step
#         if self.crossover_sampler.source_ast != current_proposal:
#             self.crossover_sampler.set_source_ast(current_proposal)

#         return super()._pre_mcmc_step(current_proposal)

    # def _generate_step_proposal(self):
    #     if self.step_index_sampled < self.step_index:
    #         self.crossover_current_step = self.rng.uniform() < self.p_crossover
    #         self.step_index_sampled = self.step_index

    #     if self.crossover_current_step:
    #         return self.crossover_sampler.sample(self.step_index, update_game_id=False)
    #     else:
    #         return super()._generate_step_proposal()


def main(args: argparse.Namespace):
    # original_recursion_limit = sys.getrecursionlimit()
    # sys.setrecursionlimit(args.recursion_limit)

    mcmc = MCMCRegrowthSampler(
        args, fitness_function_date_id=args.fitness_function_date_id,
        fitness_featurizer_path=args.fitness_featurizer_path,
        plateau_patience_steps=args.plateau_patience_steps, max_steps=args.max_steps,
        n_samples_per_step=args.n_samples_per_step, greedy_acceptance=not args.non_greedy,
        acceptance_temperature=args.acceptance_temperature,
        initial_proposal_type=InitialProposalSamplerType(args.initial_proposal_type),
        fitness_function_model_name=args.fitness_function_model_name,
        relative_path=args.relative_path, ngram_model_path=args.ngram_model_path,
    )

    if args.sample_parallel:
        initial_proposals = None
        if args.start_from_real_games:
            game_asts = list(cached_load_and_parse_games_from_file(
                os.path.join(args.relative_path, args.real_games_path), mcmc.grammar_parser,
                False, relative_path=args.relative_path))

            if args.real_game_end_index == -1:
                args.real_game_end_index = len(game_asts)

            initial_proposals = game_asts[args.real_game_start_index:min(args.real_game_end_index, len(game_asts))]

        mcmc.multiple_samples_parallel(args.n_samples, verbose=args.verbose, should_tqdm=args.should_tqdm,
                                       initial_proposals=initial_proposals, postprocess=args.postprocess,
                                       n_workers=args.parallel_n_workers, chunksize=args.parallel_chunksize)

    else:
        if args.start_from_real_games:
            game_asts = list(cached_load_and_parse_games_from_file(
                os.path.join(args.relative_path, args.real_games_path), mcmc.grammar_parser,
                False, relative_path=args.relative_path))

            for game_index in range(args.real_game_start_index, args.real_game_end_index):
                mcmc.multiple_samples(args.n_samples, verbose=args.verbose, should_tqdm=args.should_tqdm,
                                      initial_proposal=game_asts[game_index], postprocess=args.postprocess)

        else:
            mcmc.multiple_samples(args.n_samples, verbose=args.verbose,
                                  should_tqdm=args.should_tqdm, postprocess=args.postprocess)

    save_data(mcmc, args.output_folder,
              args.output_name, args.relative_path)

    # sys.setrecursionlimit(original_recursion_limit)


if __name__ == '__main__':
    args = parser.parse_args()

    args.grammar_file = os.path.join(args.relative_path, args.grammar_file)
    args.counter_output_path = os.path.join(args.relative_path, args.counter_output_path)
    args.fitness_featurizer_path = os.path.join(args.relative_path, args.fitness_featurizer_path)
    args.ngram_model_path = os.path.join(args.relative_path, args.ngram_model_path)

    args_str = '\n'.join([f'{" " * 26}{k}: {v}' for k, v in vars(args).items()])
    logger.debug(f'Shell arguments:\n{args_str}')

    main(args)
