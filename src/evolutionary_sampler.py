import argparse
from collections import OrderedDict, Counter
from datetime import datetime, timedelta
from functools import wraps
import glob
import os
import platform
import re
import signal
import shutil
import sys
import tempfile
import traceback
import typing

import logging
logging.getLogger('git').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

from git.repo import Repo
import numpy as np
import tatsu
import tatsu.ast
import tatsu.grammars
import torch
from tqdm import tqdm, trange
from viztracer import VizTracer
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import src  # type: ignore

# from ast_parser import SETUP, PREFERENCES, TERMINAL, SCORING=
import ast_printer
import ast_parser
from ast_context_fixer import ASTContextFixer
from ast_counter_sampler import *
from ast_counter_sampler import parse_or_load_counter, ASTSampler, RegrowthSampler, SamplingException, MCMC_REGRWOTH, PRIOR_COUNT, LENGTH_PRIOR
from ast_mcmc_regrowth import _load_pickle_gzip, InitialProposalSamplerType, create_initial_proposal_sampler
from ast_utils import *
from evolutionary_sampler_behavioral_features import build_behavioral_features_featurizer, BehavioralFeatureSet, BehavioralFeaturizer, DEFAULT_N_COMPONENTS
from evolutionary_sampler_diversity import *
from evolutionary_sampler_utils import Selector, UCBSelector, ThompsonSamplingSelector
from fitness_energy_utils import load_model_and_feature_columns, load_data_from_path, save_data, get_data_path, DEFAULT_SAVE_MODEL_NAME, evaluate_single_game_energy_contributions
from fitness_features import *
from fitness_ngram_models import *
from fitness_ngram_models import VARIABLE_PATTERN
from latest_model_paths import LATEST_AST_N_GRAM_MODEL_PATH, LATEST_FITNESS_FEATURIZER_PATH,\
    LATEST_FITNESS_FUNCTION_DATE_ID, LATEST_REAL_GAMES_PATH
        #LATEST_SPECIFIC_OBJECTS_AST_N_GRAM_MODEL_PATH, LATEST_SPECIFIC_OBJECTS_FITNESS_FEATURIZER_PATH, LATEST_SPECIFIC_OBJECTS_FITNESS_FUNCTION_DATE_ID


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reward-machine')))
from compile_predicate_statistics_full_database import DUCKDB_TMP_FOLDER, DUCKDB_QUERY_LOG_FOLDER  # type: ignore


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

import multiprocessing
from multiprocessing import pool as mpp
# import multiprocess as multiprocessing
# from multiprocess import pool as mpp

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


parser = argparse.ArgumentParser(description='Evolutionary Sampler')
parser.add_argument('--grammar-file', type=str, default=DEFAULT_GRAMMAR_FILE)
parser.add_argument('--parse-counter', action='store_true')
parser.add_argument('--counter-output-path', type=str, default=DEFAULT_COUNTER_OUTPUT_PATH)

parser.add_argument('--use-specific-objects-models', action='store_true')

DEFAULT_FITNESS_FUNCTION_DATE_ID = LATEST_FITNESS_FUNCTION_DATE_ID
parser.add_argument('--fitness-function-date-id', type=str, default=DEFAULT_FITNESS_FUNCTION_DATE_ID)
DEFAULT_FITNESS_FEATURIZER_PATH = LATEST_FITNESS_FEATURIZER_PATH
parser.add_argument('--fitness-featurizer-path', type=str, default=DEFAULT_FITNESS_FEATURIZER_PATH)
DEFAULT_FITNESS_FUNCTION_MODEL_NAME = DEFAULT_SAVE_MODEL_NAME
parser.add_argument('--fitness-function-model-name', type=str, default=DEFAULT_FITNESS_FUNCTION_MODEL_NAME)
parser.add_argument('--no-flip-fitness-sign', action='store_true')

DEFAULT_POPULATION_SIZE = 100
parser.add_argument('--population-size', type=int, default=DEFAULT_POPULATION_SIZE)
DEFAULT_N_STEPS = 100
parser.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS)

# TODO: rewrite these arguments to the things this sampler actually needs
# DEFAULT_PLATEAU_PATIENCE_STEPS = 1000
# parser.add_argument('--plateau-patience-steps', type=int, default=DEFAULT_PLATEAU_PATIENCE_STEPS)
# DEFAULT_MAX_STEPS = 20000
# parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS)
# DEFAULT_N_SAMPLES_PER_STEP = 1
# parser.add_argument('--n-samples-per-step', type=int, default=DEFAULT_N_SAMPLES_PER_STEP)
# parser.add_argument('--non-greedy', action='store_true')
# DEFAULT_ACCEPTANCE_TEMPERATURE = 1.0
# parser.add_argument('--acceptance-temperature', type=float, default=DEFAULT_ACCEPTANCE_TEMPERATURE)

# MICROBIAL_GA = 'microbial_ga'
# MICROBIAL_GA_WITH_BEAM_SEARCH = 'microbial_ga_with_beam_search'
# WEIGHTED_BEAM_SEARCH = 'weighted_beam_search'
MAP_ELITES = 'map_elites'
# SAMPLER_TYPES = [MICROBIAL_GA, MICROBIAL_GA_WITH_BEAM_SEARCH, WEIGHTED_BEAM_SEARCH, MAP_ELITES]
SAMPLER_TYPES = [MAP_ELITES]
parser.add_argument('--sampler-type', type=str, required=True, choices=SAMPLER_TYPES)
# parser.add_argument('--diversity-scorer-type', type=str, required=False, choices=DIVERSITY_SCORERS)
# parser.add_argument('--diversity-scorer-k', type=int, default=1)
# parser.add_argument('--diversity-score-threshold', type=float, default=0.0)
# parser.add_argument('--diversity-threshold-absolute', action='store_true')

# parser.add_argument('--microbial-ga-crossover-full-sections', action='store_true')
# parser.add_argument('--microbial-ga-crossover-type', type=int, default=2)
# DEFAULT_MICROBIAL_GA_MIN_N_CROSSOVERS = 1
# parser.add_argument('--microbial-ga-n-min-loser-crossovers', type=int, default=DEFAULT_MICROBIAL_GA_MIN_N_CROSSOVERS)
# DEFAULT_MICROBIAL_GA_MAX_N_CROSSOVERS = 5
# parser.add_argument('--microbial-ga-n-max-loser-crossovers', type=int, default=DEFAULT_MICROBIAL_GA_MAX_N_CROSSOVERS)
# DEFAULT_BEAM_SEARCH_K = 10
# parser.add_argument('--beam-search-k', type=int, default=DEFAULT_BEAM_SEARCH_K)

DEFAULT_GENERATION_SIZE = 1024
parser.add_argument('--map-elites-generation-size', type=int, default=DEFAULT_GENERATION_SIZE)
parser.add_argument('--map-elites-key-type', type=int, default=0)
parser.add_argument('--map-elites-weight-strategy', type=int, default=0)
parser.add_argument('--map-elites-initialization-strategy', type=int, default=0)
parser.add_argument('--map-elites-population-seed-path', type=str, default=None)
parser.add_argument('--map-elites-initial-candidate-pool-size', type=int, default=None)

parser.add_argument('--map-elites-use-crossover', action='store_true')
parser.add_argument('--map-elites-use-cognitive-operators', action='store_true')

features_group = parser.add_mutually_exclusive_group(required=True)
features_group.add_argument('--map-elites-behavioral-features-key', type=str, default=None)
features_group.add_argument('--map-elites-custom-behavioral-features-key', type=str, default=None,
                            choices=[feature_set_enum.value for feature_set_enum in BehavioralFeatureSet])
features_group.add_argument('--map-elites-pca-behavioral-features-indices', nargs='+', type=int, default=None)

parser.add_argument('--map-elites-pca-behavioral-features-ast-file-path', type=str, default=LATEST_REAL_GAMES_PATH)
parser.add_argument('--map-elites-pca-behavioral-features-bins-per-feature', type=int, default=None)
parser.add_argument('--map-elites-pca-behavioral-features-n-components', type=int, default=None)

parser.add_argument('--map-elites-behavioral-feature-exemplar-distance-type', type=str, default=None)
parser.add_argument('--map-elites-behavioral-feature-exemplar-distance-metric', type=str, default=None)

parser.add_argument('--map-elites-good-threshold', type=float, default=None)
parser.add_argument('--map-elites-great-threshold', type=float, default=None)

DEFAULT_RELATIVE_PATH = '.'
parser.add_argument('--relative-path', type=str, default=DEFAULT_RELATIVE_PATH)
DEFAULT_NGRAM_MODEL_PATH = LATEST_AST_N_GRAM_MODEL_PATH
parser.add_argument('--ngram-model-path', type=str, default=DEFAULT_NGRAM_MODEL_PATH)
DEFUALT_RANDOM_SEED = 33
parser.add_argument('--random-seed', type=int, default=DEFUALT_RANDOM_SEED)
parser.add_argument('--initial-proposal-type', type=int, default=0)
parser.add_argument('--sample-patience', type=int, default=100)
parser.add_argument('--sample-parallel', action='store_true')

DEFAULT_START_METHOD = 'spawn'
parser.add_argument('--parallel-start-method', type=str, default=DEFAULT_START_METHOD)
parser.add_argument('--parallel-n-workers', type=int, default=8)
parser.add_argument('--parallel-chunksize', type=int, default=1)
parser.add_argument('--parallel-maxtasksperchild', type=int, default=None)
parser.add_argument('--parallel-use-plain-map', action='store_true')
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--should-tqdm', action='store_true')
parser.add_argument('--within-step-tqdm', action='store_true')
parser.add_argument('--compute-diversity-metrics', action='store_true')
parser.add_argument('--save-interval', type=int, default=0)
parser.add_argument('--omit-rules', type=str, nargs='*')
parser.add_argument('--omit-tokens', type=str, nargs='*')
parser.add_argument('--sampler-prior-count', action='append', type=int, default=[])
parser.add_argument('--sampler-filter-func-key', type=str)
parser.add_argument('--no-weight-insert-delete-nodes-by-length', action='store_true')

DEFAULT_MAX_SAMPLE_TOTAL_SIZE = 1024 * 1024 * 5   # ~20x larger than the largest game in the real dataset
parser.add_argument('--max-sample-total-size', type=int, default=DEFAULT_MAX_SAMPLE_TOTAL_SIZE)
DEFAULT_MAX_SAMPLE_DEPTH = 16  # 24  # deeper than the deepest game, which has depth 23, and this is for a single node regrowth
parser.add_argument('--max-sample-depth', type=int, default=DEFAULT_MAX_SAMPLE_DEPTH)
DEFAULT_MAX_SAMPLE_NODES = 128  # 256  # longer than most games, but limiting a single node regrowth, not an entire game
parser.add_argument('--max-sample-nodes', type=int, default=DEFAULT_MAX_SAMPLE_NODES)

DEFAULT_OUTPUT_NAME = 'evo-sampler'
parser.add_argument('--output-name', type=str, default=DEFAULT_OUTPUT_NAME)
DEFAULT_OUTPUT_FOLDER = './samples'
parser.add_argument('--output-folder', type=str, default=DEFAULT_OUTPUT_FOLDER)

parser.add_argument('--wandb', action='store_true')
DEFAULT_WANDB_PROJECT = 'game-generation-map-elites'
parser.add_argument('--wandb-project', type=str, default=DEFAULT_WANDB_PROJECT)
DEFAULT_WANDB_ENTITY = 'guy'
parser.add_argument('--wandb-entity', type=str, default=DEFAULT_WANDB_ENTITY)

parser.add_argument('--profile', action='store_true')
parser.add_argument('--profile-output-file', type=str, default='tracer.json')
parser.add_argument('--profile-output-folder', type=str, default=tempfile.gettempdir())

parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume-max-days-back', type=int, default=1)
parser.add_argument('--start-step', type=int, default=0)


class CrossoverType(Enum):
    SAME_RULE = 0
    SAME_PARENT_INITIAL_SELECTOR = 1
    SAME_PARENT_FULL_SELECTOR = 2
    SAME_PARENT_RULE = 3
    SAME_PARENT_RULE_INITIAL_SELECTOR = 4
    SAME_PARENT_RULE_FULL_SELECTOR = 5


def _get_node_key(node: typing.Any):
    if isinstance(node, tatsu.ast.AST):
        if node.parseinfo.rule is None:  # type: ignore
            raise ValueError('Node has no rule')
        return node.parseinfo.rule  # type: ignore

    else:
        return type(node).__name__


def node_info_to_key(crossover_type: CrossoverType, node_info: ast_parser.ASTNodeInfo):
    if crossover_type == CrossoverType.SAME_RULE:
        return _get_node_key(node_info[0])

    elif crossover_type == CrossoverType.SAME_PARENT_INITIAL_SELECTOR:
        return '_'.join([_get_node_key(node_info[1]),  str(node_info[2][0])])

    elif crossover_type == CrossoverType.SAME_PARENT_FULL_SELECTOR:
        return '_'.join([_get_node_key(node_info[1]),  *[str(s) for s in node_info[2]]])

    elif crossover_type == CrossoverType.SAME_PARENT_RULE:
        return '_'.join([_get_node_key(node_info[1]), _get_node_key(node_info[0])])

    elif crossover_type == CrossoverType.SAME_PARENT_RULE_INITIAL_SELECTOR:
        return '_'.join([_get_node_key(node_info[1]),  str(node_info[2][0]),  _get_node_key(node_info[0])])

    elif crossover_type == CrossoverType.SAME_PARENT_RULE_FULL_SELECTOR:
        return '_'.join([_get_node_key(node_info[1]),  *[str(s) for s in node_info[2]],  _get_node_key(node_info[0])])

    else:
        raise ValueError(f'Invalid crossover type {crossover_type}')


ASTType: typing.TypeAlias = typing.Union[tuple, tatsu.ast.AST]
T = typing.TypeVar('T')


class SingleStepResults(typing.NamedTuple):
    samples: typing.List[ASTType]
    fitness_scores: typing.List[float]
    parent_infos: typing.List[typing.Dict[str, typing.Any]]
    diversity_scores: typing.List[float]
    sample_features: typing.List[typing.Dict[str, typing.Any]]
    operators: typing.List[str]

    def __len__(self):
        return len(self.samples)

    def accumulate(self, other: 'SingleStepResults'):
        self.samples.extend(other.samples)
        self.fitness_scores.extend(other.fitness_scores)
        if other.parent_infos is not None: self.parent_infos.extend(other.parent_infos)
        if other.diversity_scores is not None: self.diversity_scores.extend(other.diversity_scores)
        if other.sample_features is not None: self.sample_features.extend(other.sample_features)
        if other.operators is not None: self.operators.extend(other.operators)



def no_op_operator(games: typing.Union[ASTType, typing.List[ASTType]], rng=None):
    return games


def handle_multiple_inputs(operator):
    @wraps(operator)
    def wrapped_operator(self, games: typing.Union[ASTType, typing.List[ASTType]], rng: np.random.Generator, *args, **kwargs):
        if not isinstance(games, list):
            return operator(self, games, rng=rng, *args, **kwargs)

        if len(games) == 1:
            return operator(self, games[0], rng=rng, *args, **kwargs)

        else:
            operator_outputs = [operator(self, game, rng=rng, *args, **kwargs) for game in games]
            outputs = []
            for out in operator_outputs:
                if isinstance(out, list):
                    outputs.extend(out)
                else:
                    outputs.append(out)

            return outputs

    return wrapped_operator


# def msgpack_function_outputs(function):
#     @wraps(function)
#     def wrapped_function(*args, **kwargs):
#         outputs = function(*args, **kwargs)
#         return msgpack.packb(outputs)

#     return wrapped_function


PARENT_INDEX = 'parent_index'


class PopulationBasedSampler():
    args: argparse.Namespace
    candidates: SingleStepResults
    context_fixer: ASTContextFixer
    counter: ASTRuleValueCounter
    diversity_scorer: typing.Optional[DiversityScorer]
    diversity_scorer_type: typing.Optional[str]
    feature_names: typing.List[str]
    first_sampler_key: str
    fitness_featurizer: ASTFitnessFeaturizer
    fitness_featurizer_path: str
    fitness_function: typing.Callable[[torch.Tensor], float]
    fitness_function_date_id: str
    fitness_function_model_name: str
    flip_fitness_sign: bool
    generation_diversity_scores: np.ndarray
    generation_diversity_scores_index: int
    generation_index: int
    grammar: str
    grammar_parser: tatsu.grammars.Grammar  # type: ignore
    initial_samplers: typing.Dict[str, typing.Callable[[], ASTType]]
    max_sample_depth: int
    max_sample_nodes: int
    max_sample_total_size: int
    n_processes: int
    n_workers: int
    output_folder: str
    output_name: str
    postprocessor: ast_parser.ASTSamplePostprocessor
    population: typing.List[ASTType]
    population_size: int
    random_seed: int
    regrowth_sampler: RegrowthSampler
    resume: bool
    relative_path: str
    rng: np.random.Generator
    sample_filter_func: typing.Optional[typing.Callable[[ASTType, typing.Dict[str, typing.Any], float], bool]]
    sample_parallel: bool
    sampler_keys: typing.List[str]
    sampler_kwargs: typing.Dict[str, typing.Any]
    sampler_prior_count: typing.List[int]
    samplers: typing.Dict[str, ASTSampler]
    saving: bool
    signal_received: bool
    success_by_generation_and_operator: typing.List[typing.Dict[str, int]]
    verbose: int
    weight_insert_delete_nodes_by_length: bool


    '''
    This is a type of game sampler which uses an evolutionary strategy to climb a
    provided fitness function. It's a population-based alternative to the MCMC samper

    # TODO: store statistics about which locations are more likely to receive beneficial mutations?
    # TODO: keep track of 'lineages'
    '''

    def __init__(self,
                 args: argparse.Namespace,
                 population_size: int = DEFAULT_POPULATION_SIZE,
                 verbose: int = 0,
                 initial_proposal_type: InitialProposalSamplerType = InitialProposalSamplerType.MAP,
                 fitness_featurizer_path: str = DEFAULT_FITNESS_FEATURIZER_PATH,
                 fitness_function_date_id: str = DEFAULT_FITNESS_FUNCTION_DATE_ID,
                 fitness_function_model_name: str = DEFAULT_SAVE_MODEL_NAME,
                 flip_fitness_sign: bool = True,
                 relative_path: str = DEFAULT_RELATIVE_PATH,
                 output_folder: str = DEFAULT_OUTPUT_FOLDER,
                 output_name: str = DEFAULT_OUTPUT_NAME,
                 ngram_model_path: str = DEFAULT_NGRAM_MODEL_PATH,
                 sampler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 section_sampler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 sample_patience: int = 100,
                 sample_parallel: bool = False,
                 n_workers: int = 1,
                 diversity_scorer_type: typing.Optional[str] = None,
                 diversity_scorer_k: int = 1,
                 diversity_score_threshold: float = 0.0,
                 diversity_threshold_absolute: bool = False,
                 sample_filter_func: typing.Optional[typing.Callable[[ASTType, typing.Dict[str, typing.Any], float], bool]] = None,
                 sampler_prior_count: typing.List[int] = [PRIOR_COUNT],
                 weight_insert_delete_nodes_by_length: bool = True,
                 max_sample_depth: int = DEFAULT_MAX_SAMPLE_DEPTH,
                 max_sample_nodes: int = DEFAULT_MAX_SAMPLE_NODES,
                 max_sample_total_size: int = DEFAULT_MAX_SAMPLE_TOTAL_SIZE,
                 resume: bool = False,
                 ):

        self.args = args
        self.population_size = population_size
        self.verbose = verbose
        self.sample_patience = sample_patience
        self.sample_parallel = sample_parallel
        self.n_workers = n_workers
        self.n_processes = n_workers + 1  # including the main process
        self.diversity_scorer_type = diversity_scorer_type

        self.grammar = open(args.grammar_file).read()
        self.grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(self.grammar))
        self.counter = parse_or_load_counter(args, self.grammar_parser)

        self.relative_path = relative_path
        self.output_folder = output_folder
        self.output_name = output_name

        self.fitness_featurizer_path = fitness_featurizer_path
        self.fitness_featurizer = _load_pickle_gzip(fitness_featurizer_path)
        self.fitness_function_date_id = fitness_function_date_id
        self.fitness_function_model_name = fitness_function_model_name
        self.fitness_function, self.feature_names = load_model_and_feature_columns(fitness_function_date_id, name=fitness_function_model_name, relative_path=relative_path)  # type: ignore
        self.flip_fitness_sign = flip_fitness_sign

        self.diversity_scorer_type = diversity_scorer_type
        self.diversity_scorer_k = diversity_scorer_k
        self.diversity_score_threshold = diversity_score_threshold
        self.diversity_threshold_absolute = diversity_threshold_absolute

        self.diversity_scorer = None
        if self.diversity_scorer_type is not None:
            self.diversity_scorer = create_diversity_scorer(self.diversity_scorer_type, k=diversity_scorer_k, featurizer=self.fitness_featurizer, feature_names=self.feature_names)

        self.sample_filter_func = sample_filter_func
        self.sampler_prior_count = sampler_prior_count
        self.weight_insert_delete_nodes_by_length = weight_insert_delete_nodes_by_length
        self.max_sample_depth = max_sample_depth
        self.max_sample_nodes = max_sample_nodes
        self.max_sample_total_size = max_sample_total_size

        self.random_seed = args.random_seed + self._process_index()
        self.rng = np.random.default_rng(self.random_seed)

        # Used to generate the initial population of complete games
        if sampler_kwargs is None:
            sampler_kwargs = {}
        self.sampler_kwargs = sampler_kwargs

        self.samplers = {f'prior{pc}': ASTSampler(self.grammar_parser, self.counter,
                                                   max_sample_depth=self.max_sample_depth,
                                                   max_sample_nodes=self.max_sample_nodes,
                                                   seed=self.random_seed + pc,
                                                   prior_rule_count=pc, prior_token_count=pc,
                                                   length_prior={n: pc for n in LENGTH_PRIOR},
                                                   **sampler_kwargs) for pc in sampler_prior_count}
        self.sampler_keys = list(self.samplers.keys())
        self.first_sampler_key = self.sampler_keys[0]

        # Used to fix the AST context after crossover / mutation
        self.context_fixer = ASTContextFixer(self.samplers[self.first_sampler_key], rng=np.random.default_rng(self.random_seed), strict=False)

        self.initial_samplers = {  # type: ignore
            key: create_initial_proposal_sampler(initial_proposal_type, self.samplers[key], self.context_fixer,
                                                 ngram_model_path, section_sampler_kwargs)
            for key in self.sampler_keys
        }

        # Used as the mutation operator to modify existing games
        self.regrowth_sampler = RegrowthSampler(self.samplers, seed=self.random_seed, rng=np.random.default_rng(self.random_seed))

        # Initialize the candidate pools in each genera
        self.candidates = SingleStepResults([], [], [], [], [], [])

        self.postprocessor = ast_parser.ASTSamplePostprocessor()
        self.generation_index = 0
        self.fitness_metrics_history = []
        self.diversity_metrics_history = []
        self.success_by_generation_and_operator = []

        self.generation_diversity_scores = np.zeros(self.population_size)
        self.generation_diversity_scores_index = -1
        self.saving = False
        self.population_initialized = False
        self.signal_received = False
        self.resume = resume

    def initialize_population(self):
        """
        Separated to a second function that must be alled seaprately to allow for subclasses to initialize further
        """
        self.population_initialized = True

        # Do any preliminary initialization
        self._pre_population_sample_setup()

        # Generate the initial population
        self._inner_initialize_population()

        pop = self.population
        if isinstance(pop, dict):
            pop = pop.values()

        # logger.debug(f'Mean initial population_size: {np.mean([object_total_size(p) for p in pop]):.3f}')

    def _pre_population_sample_setup(self):
        pass

    def _inner_initialize_population(self):
        self.set_population([self._gen_init_sample(idx) for idx in trange(self.population_size, desc='Generating initial population')])

    def _proposal_to_features(self, proposal: ASTType) -> typing.Dict[str, typing.Any]:
        return typing.cast(dict, self.fitness_featurizer.parse(proposal, return_row=True))  # type: ignore

    def _features_to_tensor(self, features: typing.Dict[str, typing.Any]) -> torch.Tensor:
        return torch.tensor([features[name] for name in self.feature_names], dtype=torch.float32)  # type: ignore

    def _evaluate_fitness(self, features: torch.Tensor) -> float:
        fitness_function = self.fitness_function
        if 'wrapper' in fitness_function.named_steps:  # type: ignore
            fitness_function.named_steps['wrapper'].eval()  # type: ignore
        score = fitness_function.transform(features).item()
        return -score if self.flip_fitness_sign else score

    def _score_proposal(self, proposal: ASTType, return_features: bool = False):
        proposal_features = self._proposal_to_features(proposal)
        proposal_tensor = self._features_to_tensor(proposal_features)
        proposal_fitness = self._evaluate_fitness(proposal_tensor)

        if return_features:
            return proposal_fitness, proposal_features

        return proposal_fitness

    def _process_index(self):
        identity = multiprocessing.current_process()._identity  # type: ignore
        if identity is None or len(identity) == 0:
            return 0

        return identity[0] % self.n_processes

    def _sampler(self, rng: np.random.Generator) -> ASTSampler:
        return self.samplers[self._choice(self.sampler_keys, rng=rng)]  # type: ignore

    def _initial_sampler(self, rng: np.random.Generator):
        return self.initial_samplers[self._choice(self.sampler_keys, rng=rng)]  # type: ignore

    def _rename_game(self, game: ASTType, name: str) -> None:
        replace_child(game[1], ['game_name'], name)  # type: ignore

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries when spawning a new process, rather than saving
        if not self.saving:
            state['population'] = {}
            state['fitness_values'] = {}
            state['archive_cell_first_occupied'] = {}

        # Make sure we're not marking the saved model as being saved
        state['saving'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Set unique random seed per process X generation index
        self.random_seed = self.args.random_seed + (self._process_index() * (self.generation_index + 1))
        self.rng = np.random.default_rng(self.random_seed)
        self.saving = False

        for sampler_key in self.samplers:
            self.samplers[sampler_key].rng = np.random.default_rng(self.random_seed + self.samplers[sampler_key].prior_rule_count)

        self.regrowth_sampler.seed = self.random_seed
        self.regrowth_sampler.rng = np.random.default_rng(self.random_seed)
        self.context_fixer.rng = np.random.default_rng(self.random_seed)

        # trying to hard-code these as dicts to see which might be changing during iteration?!
        # self.population = dict(self.population)
        # self.fitness_values = dict(self.fitness_values)
        # self.archive_cell_first_occupied = dict(self.archive_cell_first_occupied)
        # print(f'Set state, population type: {type(self.population)} | fitness_values type: {type(self.fitness_values)} | archive_cell_first_occupied type: {type(self.archive_cell_first_occupied)}')

    def save(self, suffix: typing.Optional[str] = None, log_message: bool = True):
        self.saving = True
        output_name = self.output_name
        if suffix is not None:
            output_name += f'_{suffix}'

        save_data(self, self.output_folder, output_name, self.relative_path, log_message=log_message)
        self.saving = False

    def set_population(self, population: typing.List[typing.Any], fitness_values: typing.Optional[typing.List[float]] = None):
        '''
        Set the initial population of the sampler
        '''
        self.population = population
        self.population_size = len(population)
        if fitness_values is None:
            fitness_values = typing.cast(typing.List[float], [self._score_proposal(game, return_features=False) for game in self.population])

        self.fitness_values = fitness_values

        self.best_fitness = max(self.fitness_values)
        self.mean_fitness = np.mean(self.fitness_values)
        self.std_fitness = np.std(self.fitness_values)

        if self.diversity_scorer is not None:
            self.diversity_scorer.set_population(self.population)

    def _best_individual(self):
        return self.population[np.argmax(self.fitness_values)]

    def _print_game(self, game):
        print(ast_printer.ast_to_string(game, "\n"))

    def _choice(self, iterable: typing.Sequence[T], n: int = 1, rng: typing.Optional[np.random.Generator] = None,
                weights: typing.Optional[typing.Sequence[float]] = None) -> typing.Union[T, typing.List[T]]:
        '''
        Small hack to get around the rng invalid __array_struct__ error
        '''
        if rng is None:
            rng = self.rng

        # try:
        if n == 1:
            idx = rng.choice(len(iterable), p=weights)
            return iterable[idx]

        else:
            idxs = rng.choice(len(iterable), size=n, replace=False, p=weights)
            return [iterable[idx] for idx in idxs]

        # except ValueError as e:
        #     logger.error(f'Error in choice: len = {len(iterable)}, {n} = n, weights shape = {weights.shape}: {e}')  # type: ignore
        #     logger.error(traceback.format_exc())
        #     raise e

    def _gen_init_sample(self, idx):
        '''
        Helper function for generating an initial sample (repeating until one is generated
        without errors)
        '''
        sample = None

        while sample is None:
            try:
                sample = typing.cast(tuple, self._initial_sampler(self.rng).sample(global_context=dict(original_game_id=f'evo-{idx}')))
                if self.sample_filter_func is not None:
                    sample_fitness, sample_features = self._score_proposal(sample, return_features=True)  # type: ignore
                    if not self.sample_filter_func(sample, sample_features, sample_fitness):
                        sample = None

            except RecursionError:
                if self.verbose >= 2: logger.info(f'Recursion error in sample {idx} -- skipping')
            except SamplingException:
                if self.verbose >= 2: logger.info(f'Sampling exception in sample {idx} -- skipping')
            except ValueError:
                if self.verbose >= 2: logger.info(f'Value error in sample {idx} -- skipping')

        return sample

    def _sample_mutation(self, rng: np.random.Generator, operators: typing.Optional[typing.List[typing.Callable]] = None) -> typing.Callable[[typing.Union[ASTType, typing.List[ASTType]], np.random.Generator], typing.Union[ASTType, typing.List[ASTType]]]:
        if operators is None:
            operators = [self._gen_regrowth_sample, self._insert, self._delete]

        return self._choice(operators, rng=rng)  # type: ignore

    def _randomly_mutate_game(self, game: typing.Union[ASTType, typing.List[ASTType]], rng: np.random.Generator,
                              operators: typing.Optional[typing.List[typing.Callable]] = None) -> typing.Union[ASTType, typing.List[ASTType]]:
        return self._sample_mutation(rng, operators)(game, rng)

    @handle_multiple_inputs
    def _gen_regrowth_sample(self, game: ASTType, rng: np.random.Generator):
        # Set the source AST of the regrowth sampler to the current game
        self.regrowth_sampler.set_source_ast(game)

        return self._regrowth(rng)

    def _regrowth(self, rng: np.random.Generator, node_key_to_regrow: typing.Optional[typing.Hashable] = None) -> ASTType:
        '''
        Helper function for generating a new sample from an existing game (repeating until one is generated
        without errors)
        '''
        new_proposal = None
        sample_generated = False

        while not sample_generated:
            try:
                new_proposal = self.regrowth_sampler.sample(sample_index=0, update_game_id=False, rng=rng, node_key_to_regrow=node_key_to_regrow)
                self.context_fixer.fix_contexts(new_proposal)
                sample_generated = True

                # In this context I don't need this expensive check for identical samples, as it's just a noop
                # if ast_printer.ast_to_string(new_proposal) == ast_printer.ast_to_string(game):  # type: ignore
                #     if self.verbose >= 2: print('Regrowth generated identical games, repeating')
                # else:
                #     sample_generated = True

            except RecursionError as e:
                if self.verbose >= 2: logger.info(f'Recursion error in regrowth, skipping sample: {e.args}')
            except SamplingException as e:
                if self.verbose >= 2: logger.info(f'Sampling exception in regrowth, skipping sample: {e.args}')
            except ValueError:
                if self.verbose >= 2: logger.info(f'Value error in sample -- skipping')

        return new_proposal  # type: ignore

    def _get_valid_insert_or_delete_nodes(
            self, game: ASTType, insert: bool = True,
            weigh_nodes_by_length: bool = True, shortest_weight_maximal: bool = False, return_keys: bool = False,
            ) -> typing.Tuple[typing.List[typing.Tuple[tatsu.ast.AST, typing.List[typing.Union[str, int]], str, typing.Dict[str, typing.Any], typing.Dict[str, typing.Any]]], np.ndarray]:
        '''
        Returns a list of every node in the game which is a valid candidate for insertion or deletion
        (i.e. can have more than one child). Each entry in the list is of the form:
            (parent, selector, section, global_context, local_context)
        '''

        self.regrowth_sampler.set_source_ast(game)

        # Collect all nodes whose final selector is an integet (i.e. an index into a list) and whose parent
        # yields a list when its first selector is applied. Also make sure that the list has a minimum length
        valid_nodes = []
        for node, parent, selector, _, section, global_context, local_context in self.regrowth_sampler.parent_mapping.values():
            first_parent = parent[selector[0]]
            if isinstance(selector[-1], int) and isinstance(first_parent, list):
                parent_length = len(first_parent)
                if insert and parent_length >= 1:
                    valid_nodes.append((node, parent, selector[0], section, global_context, local_context))

                elif not insert:
                    min_length = self.samplers[self.first_sampler_key].rules[parent.parseinfo.rule][selector[0]][MIN_LENGTH]  # type: ignore
                    if parent_length >= min_length + 1:
                        valid_nodes.append((node, parent, selector[0], section, global_context, local_context))

        if len(valid_nodes) == 0:
            raise SamplingException('No valid nodes found for insertion or deletion')

        # Dedupe valid nodes based on their parent and selector
        valid_node_keys = set()
        output_valid_nodes = []
        output_node_keys = []
        output_node_weights = []
        for node, parent, selector, section, global_context, local_context in valid_nodes:
            key = (*self.regrowth_sampler._ast_key(parent), selector)
            if key not in valid_node_keys:
                valid_node_keys.add(key)
                output_valid_nodes.append((parent, selector, section, global_context, local_context))
                output_node_keys.append(self.regrowth_sampler._ast_key(node))
                output_node_weights.append(len(parent[selector]))

        if len(output_valid_nodes) > 0:
            if not weigh_nodes_by_length:
                output_node_weights = np.ones(len(output_valid_nodes)) / len(output_valid_nodes)

            else:
                output_node_weights = np.array(output_node_weights, dtype=float)
                if shortest_weight_maximal:
                    output_node_weights = output_node_weights.max() + output_node_weights.min() - output_node_weights
                output_node_weights /= output_node_weights.sum()

        else:
            output_node_weights = np.array([])

        if return_keys:
            return output_valid_nodes, output_node_weights, output_node_keys  # type: ignore

        return output_valid_nodes, output_node_weights

    @handle_multiple_inputs
    def _insert(self, game: ASTType, rng: np.random.Generator):
        '''
        Attempt to insert a new node into the provided game by identifying a node which can have multiple
        children and inserting a new node into it. The new node is selected using the initial sampler
        '''
        # Make a copy of the game
        new_game = deepcopy_ast(game)
        valid_nodes, valid_node_weights = self._get_valid_insert_or_delete_nodes(
            new_game, insert=True, weigh_nodes_by_length=self.weight_insert_delete_nodes_by_length, shortest_weight_maximal=True)

        if len(valid_nodes) == 0:
            raise SamplingException('No valid nodes found for insertion')

        # Select a random node from the list of valid nodes
        parent, selector, section, global_context, local_context = self._choice(valid_nodes, rng=rng, weights=valid_node_weights)  # type: ignore

        parent_rule = parent.parseinfo.rule # type: ignore
        parent_rule_posterior_dict = self._sampler(rng).rules[parent_rule][selector]
        assert "length_posterior" in parent_rule_posterior_dict, f"Rule {parent_rule} does not have a length posterior"

        # Sample a new rule from the parent rule posterior (parent_rule_posterior_dict['rule_posterior'])
        new_rule = posterior_dict_sample(self.rng, parent_rule_posterior_dict['rule_posterior'])

        sample_global_context = global_context.copy()  # type: ignore
        sample_global_context['rng'] = rng

        new_node = None
        while new_node is None:
            try:
                new_node = self._sampler(rng).sample(new_rule, global_context=sample_global_context, local_context=local_context) # type: ignore
            except RecursionError as e:
                if self.verbose >= 2: logger.info(f'Recursion error in insert, skipping sample: {e.args}')
            except SamplingException as e:
                if self.verbose >= 2: logger.info(f'Sampling exception in insert, skipping sample: {e.args}')
            except ValueError:
                if self.verbose >= 2: logger.info(f'Value error in sample -- skipping')

        if isinstance(new_node, tuple):
            new_node = new_node[0]

        # Insert the new node into the parent at a random index
        parent[selector].insert(rng.integers(len(parent[selector]) + 1), new_node) # type: ignore

        # Do any necessary context-fixing
        self.context_fixer.fix_contexts(new_game, crossover_child=new_node)  # type: ignore

        return new_game

    @handle_multiple_inputs
    def _delete(self, game: ASTType, rng: np.random.Generator):
        '''
        Attempt to deleting a new node into the provided game by identifying a node which can have multiple
        children and deleting one of them
        '''
        # Make a copy of the game
        new_game = deepcopy_ast(game)

        valid_nodes, valid_node_weights = self._get_valid_insert_or_delete_nodes(
            new_game, insert=False, weigh_nodes_by_length=self.weight_insert_delete_nodes_by_length, shortest_weight_maximal=False)

        if len(valid_nodes) == 0:
            raise SamplingException('No valid nodes found for deletion')

        # Select a random node from the list of valid nodes
        parent, selector, section, global_context, local_context = self._choice(valid_nodes, rng=rng, weights=valid_node_weights)  # type: ignore

        parent_rule = parent.parseinfo.rule # type: ignore
        parent_rule_posterior_dict = self._sampler(rng).rules[parent_rule][selector]
        assert "length_posterior" in parent_rule_posterior_dict, f"Rule {parent_rule} does not have a length posterior"

        # Delete a random node from the parent
        delete_index = rng.integers(len(parent[selector]))  # type: ignore
        child_to_delete = parent[selector][delete_index]  # type: ignore

        del parent[selector][delete_index] # type: ignore

        # Do any necessary context-fixing
        self.context_fixer.fix_contexts(new_game, original_child=child_to_delete)  # type: ignore

        return new_game

    def _crossover(self, games: typing.Union[ASTType, typing.List[ASTType]],
                   rng: typing.Optional[np.random.Generator] = None,
                   crossover_type: typing.Optional[CrossoverType] = None,
                   crossover_first_game: bool = True, crossover_second_game: bool = True):
        '''
        Attempts to perform a crossover between the two given games. The crossover type determines
        how nodes in the game are categorized (i.e. by rule, by parent rule, etc.). The crossover
        is performed by finding the set of 'categories' that are present in both games, and then
        selecting a random category from which to sample the nodes that will be exchanged. If no
        categories are shared between the two games, then no crossover is performed
        '''
        if not crossover_first_game and not crossover_second_game:
            raise ValueError("At least one of crossover_first_game and crossover_second_game must be True")

        if rng is None:
            rng = self.rng

        if crossover_type is None:
            crossover_type = typing.cast(CrossoverType, self._choice(list(CrossoverType), rng=rng))

        game_2 = None
        if isinstance(games, list):
            game_1 = games[0]

            if len(games) > 1:
                game_2 = games[1]
        else:
            game_1 = games

        if game_2 is None:
            game_2 = typing.cast(ASTType, self._choice(self.population, rng=rng))

        if crossover_first_game:
            game_1 = deepcopy_ast(game_1)

        if crossover_second_game:
            game_2 = deepcopy_ast(game_2)

        # Create a map from crossover_type keys to lists of nodeinfos for each game
        self.regrowth_sampler.set_source_ast(game_1)
        game_1_crossover_map = defaultdict(list)
        for node_key in self.regrowth_sampler.node_keys:
            node_info = self.regrowth_sampler.parent_mapping[node_key]
            game_1_crossover_map[node_info_to_key(crossover_type, node_info)].append(node_info)

        self.regrowth_sampler.set_source_ast(game_2)
        game_2_crossover_map = defaultdict(list)
        for node_key in self.regrowth_sampler.node_keys:
            node_info = self.regrowth_sampler.parent_mapping[node_key]
            game_2_crossover_map[node_info_to_key(crossover_type, node_info)].append(node_info)

        # Find the set of crossover_type keys that are shared between the two games
        shared_crossover_keys = set(game_1_crossover_map.keys()).intersection(set(game_2_crossover_map.keys()))

        # If there are no shared crossover keys, then throw an exception
        if len(shared_crossover_keys) == 0:
            raise SamplingException("No crossover keys shared between the two games")

        # Select a random crossover key and a nodeinfo for each game with that key
        crossover_key = self._choice(list(shared_crossover_keys), rng=rng)
        game_1_selected_node_info = self._choice(game_1_crossover_map[crossover_key], rng=rng)
        game_2_selected_node_info = self._choice(game_2_crossover_map[crossover_key], rng=rng)

        # Create new copies of the nodes to be crossed over
        g1_node, g1_parent, g1_selector = game_1_selected_node_info[:3]
        g2_node, g2_parent, g2_selector = game_2_selected_node_info[:3]

        # Perform the crossover and fix the contexts of the new games
        if crossover_first_game:
            game_2_crossover_node = deepcopy_ast(g2_node, copy_type=ASTCopyType.NODE)
            replace_child(g1_parent, g1_selector, game_2_crossover_node) # type: ignore
            self.context_fixer.fix_contexts(game_1, g1_node, game_2_crossover_node)  # type: ignore

        if crossover_second_game:
            game_1_crossover_node = deepcopy_ast(g1_node, copy_type=ASTCopyType.NODE)
            replace_child(g2_parent, g2_selector, game_1_crossover_node) # type: ignore
            self.context_fixer.fix_contexts(game_2, g2_node, game_1_crossover_node)  # type: ignore

        return [game_1, game_2]

    def _crossover_insert(self, games: typing.Union[ASTType, typing.List[ASTType]],
                   rng: typing.Optional[np.random.Generator] = None,
                   crossover_first_game: bool = True, crossover_second_game: bool = True):

        if rng is None:
            rng = self.rng

        crossover_type = CrossoverType.SAME_PARENT_INITIAL_SELECTOR

        game_2 = None
        if isinstance(games, list):
            game_1 = games[0]

            if len(games) > 1:
                game_2 = games[1]
        else:
            game_1 = games

        if game_2 is None:
            game_2 = typing.cast(ASTType, self._choice(self.population, rng=rng))

        if crossover_first_game:
            game_1 = deepcopy_ast(game_1)

        if crossover_second_game:
            game_2 = deepcopy_ast(game_2)

        # Create a map from crossover_type keys to lists of nodeinfos for each game
        _, _, game_1_insertion_node_keys = self._get_valid_insert_or_delete_nodes(  # type: ignore
            game_1, insert=True, weigh_nodes_by_length=self.weight_insert_delete_nodes_by_length,
            shortest_weight_maximal=True, return_keys=True)

        game_1_crossover_map = defaultdict(list)
        for node_key in game_1_insertion_node_keys:
            node_info = self.regrowth_sampler.parent_mapping[node_key]
            game_1_crossover_map[node_info_to_key(crossover_type, node_info)].append(node_info)

        _, _, game_2_insertion_node_keys = self._get_valid_insert_or_delete_nodes(  # type: ignore
            game_2, insert=True, weigh_nodes_by_length=self.weight_insert_delete_nodes_by_length,
            shortest_weight_maximal=True, return_keys=True)

        game_2_crossover_map = defaultdict(list)
        for node_key in game_2_insertion_node_keys:
            node_info = self.regrowth_sampler.parent_mapping[node_key]
            game_2_crossover_map[node_info_to_key(crossover_type, node_info)].append(node_info)

        # Find the set of crossover_type keys that are shared between the two games
        shared_crossover_keys = set(game_1_crossover_map.keys()).intersection(set(game_2_crossover_map.keys()))

        # If there are no shared crossover keys, then throw an exception
        if len(shared_crossover_keys) == 0:
            raise SamplingException("No insertion-friendly crossover keys shared between the two games")

        # Select a random crossover key and a nodeinfo for each game with that key
        crossover_key = self._choice(list(shared_crossover_keys), rng=rng)
        # print('Crossover key', crossover_key)
        game_1_selected_node_info = self._choice(game_1_crossover_map[crossover_key], rng=rng)
        game_2_selected_node_info = self._choice(game_2_crossover_map[crossover_key], rng=rng)

        # Create new copies of the nodes to be crossed over
        g1_node, g1_parent, g1_selector = game_1_selected_node_info[:3]
        g2_node, g2_parent, g2_selector = game_2_selected_node_info[:3]

        # Insert the new node into the parent at a random index
        if crossover_first_game:
            game_2_crossover_node = deepcopy_ast(g2_node, copy_type=ASTCopyType.NODE)
            g1_parent[g1_selector[0]].insert(rng.integers(len(g1_parent[g1_selector[0]]) + 1), game_2_crossover_node)
            self.context_fixer.fix_contexts(game_1, crossover_child=game_2_crossover_node)  # type: ignore

        if crossover_second_game:
            game_1_crossover_node = deepcopy_ast(g1_node, copy_type=ASTCopyType.NODE)
            g2_parent[g2_selector[0]].insert(rng.integers(len(g2_parent[g2_selector[0]]) + 1), game_1_crossover_node)
            self.context_fixer.fix_contexts(game_2, crossover_child=game_1_crossover_node)  # type: ignore

        return [game_1, game_2]

    def _find_index_for_section(self, existing_sections: typing.List[str], new_section: str) -> typing.Tuple[int, bool]:
        try:
            index = existing_sections.index(new_section)
            return index, True

        except ValueError:
            if new_section == ast_parser.SETUP:
                return 0, False

            # in this case, it's ast_parser.TERMINAL
            return len(existing_sections) - 1, False

    def _insert_section_to_game(self, game: ASTType, new_section: tuple, index: int, replace: bool):
        continue_index = index if not replace else index + 1
        return (*game[:index], new_section, *game[continue_index:])  # type: ignore

    def _remove_section_from_game(self, game: ASTType, index: int):
        return (*game[:index], *game[index + 1:])  # type: ignore

    def _crossover_full_sections(self, games: typing.Union[ASTType, typing.List[ASTType]], rng: typing.Optional[np.random.Generator] = None,
                                 crossover_first_game: bool = True, crossover_second_game: bool = True):

        if not crossover_first_game and not crossover_second_game:
            raise ValueError("At least one of crossover_first_game and crossover_second_game must be True")

        if rng is None:
            rng = self.rng

        game_2 = None
        if isinstance(games, list):
            game_1 = games[0]

            if len(games) > 1:
                game_2 = games[1]
        else:
            game_1 = games

        if game_2 is None:
            game_2 = typing.cast(ASTType, self._choice(self.population, rng=rng))

        if crossover_first_game:
            game_1 = deepcopy_ast(game_1)

        if crossover_second_game:
            game_2 = deepcopy_ast(game_2)

        game_1_sections = [t[0] for t in game_1[3:-1]]  # type: ignore
        game_2_sections = [t[0] for t in game_2[3:-1]]  # type: ignore

        if crossover_first_game:
            game_2_section_index = rng.integers(len(game_2_sections))
            game_2_section = game_2_sections[game_2_section_index]
            index, replace = self._find_index_for_section(game_1_sections, game_2_section)
            section_copy = deepcopy_ast(game_2[3 + game_2_section_index], copy_type=ASTCopyType.SECTION)
            self._insert_section_to_game(game_1, section_copy, index, replace)  # type: ignore
            self.context_fixer.fix_contexts(game_1, crossover_child=section_copy[1])  # type: ignore

        if crossover_second_game:
            game_1_section_index = rng.integers(len(game_1_sections))
            game_1_section = game_1_sections[game_1_section_index]
            index, replace = self._find_index_for_section(game_2_sections, game_1_section)
            section_copy = deepcopy_ast(game_1[3 + game_1_section_index], copy_type=ASTCopyType.SECTION)
            self._insert_section_to_game(game_2, section_copy, index, replace)  # type: ignore
            self.context_fixer.fix_contexts(game_2, crossover_child=section_copy[1])  # type: ignore

        return [game_1, game_2]

    def _resample_variable_types(self, game: ASTType, rng: np.random.Generator):
        return self._cognitive_inspired_mutate_preference(game, rng, resample_variable_types=True)

    def _resample_first_condition(self, game: ASTType, rng: np.random.Generator):
        return self._cognitive_inspired_mutate_preference(game, rng, mutate_first_condition=True)

    def _resample_last_condition(self, game: ASTType, rng: np.random.Generator):
        return self._cognitive_inspired_mutate_preference(game, rng, mutate_last_condition=True)

    def _resample_variable_types_and_first_condition(self, game: ASTType, rng: np.random.Generator):
        return self._cognitive_inspired_mutate_preference(game, rng, mutate_first_condition=True, resample_variable_types=True)

    def _resample_variable_types_and_last_condition(self, game: ASTType, rng: np.random.Generator):
        return self._cognitive_inspired_mutate_preference(game, rng, mutate_last_condition=True, resample_variable_types=True)

    @handle_multiple_inputs
    def _cognitive_inspired_mutate_preference(self, game: ASTType, rng: np.random.Generator, mutate_first_condition: bool = False,
                                    mutate_last_condition: bool = False, resample_variable_types: bool = False,
                                    sample_additional_variable: typing.Optional[bool] = None,
                                    mutated_preference_as_new: typing.Optional[bool] = None) -> ASTType:
        game = deepcopy_ast(game)  # type: ignore

        if mutated_preference_as_new is None:
            mutated_preference_as_new = rng.uniform() < 0.5

        if sample_additional_variable is None:
            sample_additional_variable = rng.uniform() < 0.5

        # if we're adding the preference, we should check that the preferences are a list
        preferences_node = None
        if mutated_preference_as_new:
            preferences_node = [section_tuple for section_tuple in game if section_tuple[0] == ast_parser.PREFERENCES][0][1]

            if isinstance(preferences_node.preferences, tatsu.ast.AST):
                replace_child(preferences_node, ['preferences'], [preferences_node.preferences])  # type: ignore

        self.regrowth_sampler.set_source_ast(game)
        pref_def_node_keys = []
        for node_key in self.regrowth_sampler.node_keys:
            node_info = self.regrowth_sampler.parent_mapping[node_key]
            if node_info[0].parseinfo.rule == 'pref_def':  # type: ignore
                pref_def_node_keys.append(node_key)

        if len(pref_def_node_keys) == 0:
            raise SamplingException('No preference nodes found for mutation')

        pref_def_node_key = self._choice(pref_def_node_keys, rng=rng)
        pref_def_node = self.regrowth_sampler.parent_mapping[pref_def_node_key][0]  # type: ignore

        # if we're adding the mutated preference, we should keep a copy of the original
        original_pref_def = None
        if mutated_preference_as_new:
            original_pref_def = deepcopy_ast(pref_def_node, copy_type=ASTCopyType.NODE)

        if pref_def_node.definition.parseinfo.rule == 'pref_forall':  # type: ignore
            pref_forall_pref = pref_def_node.definition.forall_pref.preferences  # type: ignore
            if not isinstance(pref_forall_pref, tatsu.ast.AST):
                pref_forall_pref = self._choice(pref_forall_pref, rng=rng)

            pref_body = pref_forall_pref.pref_body  # type: ignore

        else:
            pref_body = pref_def_node.definition.pref_body.body  # type: ignore

        pref_body = typing.cast(tatsu.ast.AST, pref_body)

        if pref_body.parseinfo.rule == 'pref_body_exists':  # type: ignore
            variables = pref_body.exists_vars.variables  # type: ignore

            if resample_variable_types:
                if isinstance(variables, tatsu.ast.AST):
                    type_def_node = variables.var_type
                    type_def_node_key = self.regrowth_sampler._ast_key(type_def_node)
                    global_context, local_context = self.regrowth_sampler.parent_mapping[type_def_node_key][-2:]
                    rule = typing.cast(str, type_def_node.parseinfo.rule)  # type: ignore
                    replace_child(variables, ['var_type'], self._sampler(rng).sample(rule, global_context, local_context)[0])

                else:
                    variables_to_resample = rng.uniform(size=len(variables)) < 0.5
                    if not variables_to_resample.any():
                        variables_to_resample[rng.integers(len(variables))] = True

                    for i, resample in enumerate(variables_to_resample):
                        if resample:
                            type_def_node = variables[i].var_type
                            type_def_node_key = self.regrowth_sampler._ast_key(type_def_node)
                            global_context, local_context = self.regrowth_sampler.parent_mapping[type_def_node_key][-2:]
                            replace_child(variables[i], ['var_type'], self._sampler(rng).sample(type_def_node.parseinfo.rule, global_context, local_context)[0])

            if sample_additional_variable:
                # before_str = ast_printer.ast_section_to_string(pref_body.exists_vars, ast_parser.PREFERENCES)
                node_key = self.regrowth_sampler._ast_key(pref_body.exists_vars)
                sampler = self._sampler(rng)
                varible_def_rules, variable_def_probs = zip(*sampler.rules['variable_list']['variables'][RULE_POSTERIOR].items())
                new_variable_def_rule = typing.cast(str, self._choice(varible_def_rules, weights=variable_def_probs, rng=rng))
                global_context, local_context = self.regrowth_sampler.parent_mapping[node_key][-2:]

                current_variable_types = set()
                if isinstance(variables, tatsu.ast.AST):
                    current_variable_types.update(ast_parser._extract_variable_type_as_list(variables.var_type.type))

                else:
                    for var_def in variables:
                        current_variable_types.update(ast_parser._extract_variable_type_as_list(var_def.var_type.type))

                new_variable_node = None
                new_variable_node_has_new_type = False
                while not new_variable_node_has_new_type:
                    new_variable_node = sampler.sample(new_variable_def_rule, global_context, local_context)[0]
                    new_variable_node_types = set(ast_parser._extract_variable_type_as_list(new_variable_node.var_type.type))
                    new_variable_node_has_new_type = len(new_variable_node_types.difference(current_variable_types)) > 0

                if isinstance(variables, tatsu.ast.AST):
                    new_variable_list = [variables, new_variable_node]
                    replace_child(pref_body.exists_vars, 'variables', new_variable_list)  # type: ignore

                else:
                    variables.append(new_variable_node)

                # after_str = ast_printer.ast_section_to_string(pref_body.exists_vars, ast_parser.PREFERENCES)
                # print(f'Added variable to preference: {before_str} -> {after_str}')

            pref_body = typing.cast(tatsu.ast.AST, pref_body.exists_args)

        if mutate_first_condition or mutate_last_condition:
            if pref_body.parseinfo.rule == 'then':  # type: ignore
                seq_funcs = typing.cast(list, pref_body.then_funcs)
                seq_func = seq_funcs[0 if mutate_first_condition else -1].seq_func
                if seq_func.parseinfo.rule == 'once':
                    pred = seq_func.once_pred

                elif seq_func.parseinfo.rule == 'once_measure':
                    pred = seq_func.once_measure_pred

                elif seq_func.parseinfo.rule in ('hold', 'while_hold'):
                    pred = seq_func.hold_pred

                else:
                    raise ValueError(f'Unexpected sequence function rule: {seq_func.parseinfo.rule}')

            elif pref_body.parseinfo.rule == 'at_end':  # type: ignore
                pred = pref_body.at_end_pred

            else:
                raise ValueError(f'Unexpected preference body rule: {pref_body.parseinfo.rule}')   # type: ignore

            pred_key = self.regrowth_sampler._ast_key(pred)

            mutated_game = self._regrowth(rng, node_key_to_regrow=pred_key)
            game = mutated_game

        if mutated_preference_as_new:
            game_preferences_node = [section_tuple for section_tuple in game if section_tuple[0] == ast_parser.PREFERENCES][0][1]
            game_preferences_node['preferences'].insert(rng.integers(len(game_preferences_node['preferences']) + 1), original_pref_def)  # type: ignore

        self.context_fixer.fix_contexts(game)  # type: ignore

        return game

    @handle_multiple_inputs
    def _sample_or_resample_setup(self, game: ASTType, rng: np.random.Generator, p_delete_existing_section: float = 0.5):
        new_game = deepcopy_ast(game)
        game_has_setup = new_game[3][0] == ast_parser.SETUP  # type: ignore

        if game_has_setup and rng.uniform() < p_delete_existing_section:
            new_game = self._remove_section_from_game(new_game, 3)

        else:
            new_setup = None
            global_context = dict(rng=rng)
            while new_setup is None:
                try:
                    new_setup = self._sampler(rng).sample('setup', global_context=global_context)[0]
                except SamplingException as e:
                    if self.verbose > 1:
                        logger.info(f'Failed to sample setup with global context {global_context}: {e.args}')
                    continue

            new_setup_tuple = (ast_parser.SETUP, new_setup, ')')
            new_game = self._insert_section_to_game(new_game, new_setup_tuple, 3, replace=new_game[3][0] == ast_parser.SETUP)  # type: ignore

        self.context_fixer.fix_contexts(new_game)  # type: ignore
        return new_game

    @handle_multiple_inputs
    def _sample_or_resample_terminal(self, game: ASTType, rng: np.random.Generator, p_delete_existing_section: float = 0.5):
        new_game = deepcopy_ast(game)
        game_has_terminal = new_game[-3][0] == ast_parser.TERMINAL  # type: ignore

        if game_has_terminal and rng.uniform() < p_delete_existing_section:
            new_game = self._remove_section_from_game(new_game, len(new_game) - 3)

        else:
            new_terminal = None

            base_scoring_node = game[-2][1]  # type: ignore
            self.regrowth_sampler.set_source_ast(game)
            base_scoring_node_key = self.regrowth_sampler._ast_key(base_scoring_node)
            global_context = self.regrowth_sampler.parent_mapping[base_scoring_node_key][-2]  # type: ignore
            global_context['rng'] = rng

            while new_terminal is None:
                try:
                    new_terminal = self._sampler(rng).sample('terminal', global_context=global_context)[0]
                except SamplingException as e:
                    if self.verbose > 1:
                        logger.info(f'Failed to sample terminal with global context {global_context}: {e.args}')
                    continue

            new_terminal_tuple = (ast_parser.TERMINAL, new_terminal, ')')

            index = len(new_game) - 2 if not game_has_terminal else len(new_game) - 3
            new_game = self._insert_section_to_game(new_game, new_terminal_tuple, index, replace=game_has_terminal)  # type: ignore

        self.context_fixer.fix_contexts(new_game)  # type: ignore
        return new_game

    @handle_multiple_inputs
    def _resample_scoring(self, game: ASTType, rng: np.random.Generator):
        new_game = deepcopy_ast(game)
        base_scoring_node = game[-2][1]  # type: ignore
        new_scoring = None

        self.regrowth_sampler.set_source_ast(game)
        base_scoring_node_key = self.regrowth_sampler._ast_key(base_scoring_node)
        global_context = self.regrowth_sampler.parent_mapping[base_scoring_node_key][-2]  # type: ignore
        global_context['rng'] = rng

        while new_scoring is None:
            try:
                new_scoring = self._sampler(rng).sample('scoring_expr', global_context=global_context)[0]
            except SamplingException as e:
                if self.verbose > 1:
                    logger.info(f'Failed to sample terminal with global context {global_context}: {e.args}')
                continue

        new_scoring_tuple = (ast_parser.SCORING, new_scoring, ')')

        index = len(new_game) - 2
        new_game = self._insert_section_to_game(new_game, new_scoring_tuple, index, replace=True)  # type: ignore
        return new_game

    def _get_operator(self, rng: typing.Optional[np.random.Generator] = None) -> typing.Callable[[typing.Union[ASTType, typing.List[ASTType]], np.random.Generator], typing.Union[ASTType, typing.List[ASTType]]]:
        '''
        Returns a function (operator) which takes in a list of games and returns a list of new games.
        As a default, always return a no_op operator
        '''

        return no_op_operator

    def _get_parent_iterator(self, n_parents_per_sample: int = 1, n_times_each_parent: int = 1) -> typing.Iterator[typing.Tuple[typing.Union[ASTType, typing.List[ASTType]], typing.Dict[str, typing.Any]]]:
        '''
        Returns an iterator which at each step yields one or more parents that will be modified
        by the operator. As a default, return an iterator which yields the entire population
        '''
        indices = np.concatenate([self.rng.permutation(self.population_size) for _ in range(n_times_each_parent)])

        if n_parents_per_sample == 1:
            for i in indices:
                yield (self.population[i], {PARENT_INDEX: i})

        else:
            for idxs in range(0, len(indices), n_parents_per_sample):
                sample_indices = indices[idxs:idxs + n_parents_per_sample]
                yield ([self.population[i] for i in sample_indices], {PARENT_INDEX: sample_indices})

    def _update_generation_diversity_scores(self):
        if self.diversity_scorer is not None and self.generation_index != self.generation_diversity_scores_index:
            if self.verbose:
                logger.info(f'Updating diversity scores for generation {self.generation_index}')

            population_diversity_scores = self.diversity_scorer.population_score_distribution()
            self.generation_diversity_scores = population_diversity_scores
            self.generation_diversity_scores_index = self.generation_index

    def _end_single_evolutionary_step(self, results: typing.Optional[SingleStepResults] = None):
        '''
        Returns the new population given the current population, the candidate games, and the
        scores for both the population and the candidate games. As a default, return the top P
        games from the union of the population and the candidates
        '''
        if results is None:
            results = self.candidates

        candidates = results.samples
        candidate_scores = results.fitness_scores
        # parent_infos = results.parent_infos
        candidate_diversity_scores = results.diversity_scores
        # candidate_features = results.sample_features

        if candidate_diversity_scores is not None and len(candidate_diversity_scores) > 0:
            diversity_scores = np.array(candidate_diversity_scores)  # type: ignore

            if self.verbose:
                logger.info(f'Candidate diversity scores: min: {diversity_scores.min():.3f}, 25th percentile: {np.percentile(diversity_scores, 25):.3f} mean: {diversity_scores.mean():.3f}, 75th percentile: {np.percentile(diversity_scores, 25):.3f},  max: {diversity_scores.max():.3f},')

            if not self.diversity_threshold_absolute:
                self._update_generation_diversity_scores()
                threshold = np.percentile(self.generation_diversity_scores, self.diversity_score_threshold)
                if self.verbose:
                    logger.info(f'Using diversity threshold of {threshold} (percentile {self.diversity_score_threshold} of generation {self.generation_index} diversity scores, min {self.generation_diversity_scores.min()}, max {self.generation_diversity_scores.max()})')
            else:
                threshold = self.diversity_score_threshold

            diverse_candidate_indices = np.where(diversity_scores >= threshold)[0]
            if len(diverse_candidate_indices) == 0:
                logger.warning(f'No diverse candidates found with a threshold of {threshold} (highest candidate diversity score was {diversity_scores.max()}), not replacing any population members')
                return

            diversity_message = ''
            if self.verbose:
                diversity_message = f'Found {len(diverse_candidate_indices)} diverse candidates (highest candidate diversity score was {diversity_scores.max()}'

            candidates = [candidates[i] for i in diverse_candidate_indices]
            candidate_scores = [candidate_scores[i] for i in diverse_candidate_indices]

            if self.verbose:
                diversity_message += f', highest diverse candidate fitness score was {max(candidate_scores)})'
                logger.info(diversity_message)

        all_games = self.population + candidates
        all_scores = self.fitness_values + candidate_scores

        top_indices = np.argsort(all_scores)[-self.population_size:]
        self.set_population([all_games[i] for i in top_indices], [all_scores[i] for i in top_indices])

    def _sample_and_apply_operator(self, parent: typing.Union[ASTType, typing.List[ASTType]],
                                   parent_info: typing.Dict[str, typing.Any],
                                   generation_index: int, sample_index: int,
                                   return_sample_features: bool = False) -> SingleStepResults:
        '''
        Given a parent, a generation and sample index (to make sure that the RNG is seeded differently for each generation / individual),
        sample an operator and apply it to the parent. Returns the child or children and their fitness scores
        '''
        rng = np.random.default_rng(self.random_seed + (self.population_size * generation_index) + sample_index)  # type: ignore
        compute_features = return_sample_features or self.sample_filter_func is not None

        for _ in range(self.sample_patience):
            try:
                operator = self._get_operator(rng)
                child_or_children = operator(parent, rng)
                if not isinstance(child_or_children, list):
                    child_or_children = [child_or_children]

                children = []
                children_fitness_scores = []
                children_features = []
                operators = []

                for i, child in enumerate(child_or_children):
                    # child_size = object_total_size(child)
                    # if child_size > self.max_sample_total_size:
                    #     # TODO: move this to be only if verbose at some point in the future
                    #     parent_size = [object_total_size(p) for p in parent] if isinstance(parent, list) else object_total_size(parent)
                    #     section_sizes = [object_total_size(s) for s in child[3:-1]]  # type: ignore
                    #     logger.info(f'Sample size {child_size} ({section_sizes}) exceeds max size {self.max_sample_total_size} from parent with size {parent_size}, skipping')
                    #     continue

                    self._rename_game(child, f'evo-{generation_index}-{sample_index}-{i}')
                    child = typing.cast(ASTType, self.postprocessor(child, should_deepcopy_initial=False))
                    retval = self._score_proposal(child, return_features=compute_features)
                    if compute_features:
                        fitness, features = retval  # type: ignore
                    else:
                        fitness, features = retval, None

                    if self.sample_filter_func is not None and not self.sample_filter_func(child, features, fitness):  # type: ignore
                        continue

                    children.append(child)
                    children_fitness_scores.append(fitness)
                    children_features.append(features)
                    operators.append(operator.__name__)

                if len(children) == 0:
                    raise SamplingException('No children passed the filter func')

                children_features = None if not return_sample_features else children_features
                children_diversity_scores = [self.diversity_scorer(child) for child in children] if self.diversity_scorer is not None else None

                if not isinstance(parent_info, list) or len(parent_info) != len(children):
                    parent_info = itertools.repeat(parent_info)  # type: ignore

                return SingleStepResults(children, children_fitness_scores, parent_info, children_diversity_scores, children_features, operators)  # type: ignore

            except SamplingException as e:
                # if self.verbose:
                #     logger.info(f'Could not validly sample an operator and apply it to a child, retrying: {e}')
                continue
            except RecursionError as e:
                # if self.verbose:
                #     logger.info(f'Could not validly sample an operator and apply it to a child, retrying: {e}')
                continue
            except ValueError as e:
                # logging.error(traceback.format_exc())
                # raise e
                continue
            except RuntimeError as e:
                logging.error(traceback.format_exc())
                raise e
            except Exception as e:
                logging.error(f'Unexpected error in _sample_and_apply_operator: {e}')
                logging.error(traceback.format_exc())
                raise e


        # Parent is already in the population, so returning nothing
        # raise SamplingException(f'Could not validly sample an operator and apply it to a child in {self.sample_patience} attempts')
        return SingleStepResults([], [], [], [], [], [])

    def _sample_and_apply_operator_map_wrapper(self, args):
        """
        Here to enable adding other arguments to the parent iterator and param iterator, and
        to not have ot rely on implementations of starmap existing
        """
        try:
            return self._sample_and_apply_operator(*args[0], *args[1:])
        except Exception as e:
            logger.error(f'Exception caught in _sample_and_apply_operator_map_wrapper: {e}')
            logger.error(traceback.format_exc())
            raise e

    def _build_evolutionary_step_param_iterator(self, parent_iterator: typing.Iterable[typing.Tuple[typing.Union[ASTType, typing.List[ASTType]], typing.Optional[typing.Dict[str, typing.Any]]]]) -> typing.Iterable[typing.Tuple[typing.Union[ASTType, typing.List[ASTType]], int, int]]:
        '''
        Given an iterator over parents, return an iterator over tuples of (parent, generation_index, sample_index)
        '''
        return zip(parent_iterator, itertools.repeat(self.generation_index), itertools.count())

    def evolutionary_step(self, pool: typing.Optional[mpp.Pool] = None, chunksize: int = 1,
                          should_tqdm: bool = False, use_imap: bool = True):
        # The core steps are:
        # 1. determine which "operator" is going to be used (an operator takes in one or more games and produces one or more new games)
        # 2. create a "parent_iteraor" which takes in the population and yields the parents that will be used by the operator
        # 3. for each parent(s) yielded, apply the operator to produce one or more new games and add them to a "candidates" list
        # 4. score the candidates
        # 5. pass the initial population and the candidates to the "selector" which will return the new population

        if hasattr(self, 'population_initialized') and not self.population_initialized:
            raise ValueError('Cannot run evolutionary steps in parallel without initializing the population first')

        param_iterator = self._build_evolutionary_step_param_iterator(self._get_parent_iterator())

        if pool is not None:
            if use_imap:
                children_iter = pool.imap_unordered(self._sample_and_apply_operator_map_wrapper, param_iterator, chunksize=chunksize)  # type: ignore
            else:
                children_iter = pool.map(self._sample_and_apply_operator_map_wrapper, param_iterator, chunksize=chunksize)  # type: ignore
        else:
            children_iter = map(self._sample_and_apply_operator_map_wrapper, param_iterator)  # type: ignore

        if should_tqdm:
            children_iter = tqdm(children_iter)  # type: ignore

        try:
            for step_results in children_iter:
                self._handle_single_operator_results(step_results)
        except Exception as e:
            logger.error(f'Exception caught in evolutionary_step: {e}')
            logger.error(traceback.format_exc())
            raise e

        self._end_single_evolutionary_step()

    def _handle_single_operator_results(self, results: SingleStepResults):
        self.candidates.accumulate(results)

    def _create_tqdm_postfix(self) -> typing.Dict[str, str]:
        baseline_postfix = {"Mean": f"{self.mean_fitness:.2f}", "Std": f"{self.std_fitness:.2f}", "Max": f"{self.best_fitness:.2f}"}
        baseline_postfix.update(self._custom_tqdm_postfix())
        return baseline_postfix

    def _custom_tqdm_postfix(self) -> typing.Dict[str, str]:
        return {
            "DivMean": f"{self.diversity_metrics_history[-1]['mean']:.2f}",
            "DivStd": f"{self.diversity_metrics_history[-1]['std']:.2f}",
            "DivMax": f"{self.diversity_metrics_history[-1]['max']:.2f}",
            "DivMin": f"{self.diversity_metrics_history[-1]['min']:.2f}",
        }

    def multiple_evolutionary_steps(self, total_steps: int, pool: typing.Optional[mpp.Pool] = None,
                                    chunksize: int = 1, should_tqdm: bool = False, inner_tqdm: bool = False,
                                    use_imap: bool = True, start_step: int = 0,
                                    compute_diversity_metrics: bool = False, save_interval: int = 0):

        if hasattr(self, 'population_initialized') and not self.population_initialized:
            raise ValueError('Cannot run multiple evolutionary steps without initializing the population first')

        step_iter = range(start_step, total_steps)
        if should_tqdm:
            pbar = tqdm(total=total_steps, desc="Evolutionary steps") # type: ignore
            if start_step > 0:
                pbar.update(start_step)

        for _ in step_iter:  # type: ignore
            self.success_by_generation_and_operator.append(defaultdict(int))
            self.evolutionary_step(pool, chunksize, should_tqdm=inner_tqdm, use_imap=use_imap)

            if compute_diversity_metrics:
                if self.diversity_scorer is None:
                    raise ValueError('Cannot compute diversity metrics without a diversity scorer')

                self._update_generation_diversity_scores()
                self.diversity_metrics_history.append({
                    'mean': self.generation_diversity_scores.mean(),
                    'std': self.generation_diversity_scores.std(),
                    'max': self.generation_diversity_scores.max(),
                    'min': self.generation_diversity_scores.min()
                })

            if should_tqdm or args.wandb:
                postfix = self._create_tqdm_postfix()
                if should_tqdm:
                    pbar.update(1)  # type: ignore
                    pbar.set_postfix(postfix)  # type: ignore

                if args.wandb:
                    wandb_update = {k: float(v) if isinstance(v, str) else v for k, v in postfix.items() if k != 'Timestamp'}
                    wandb_update['step'] = self.generation_index
                    wandb.log(wandb_update)

            elif self.verbose:
                logger.info(f"Average fitness: {self.mean_fitness:.2f} +/- {self.std_fitness:.2f}, Best fitness: {self.best_fitness:.2f}")

            self.fitness_metrics_history.append({'mean': self.mean_fitness, 'std': self.std_fitness, 'max': self.best_fitness})

            self.generation_index += 1

            # This is required because the changes to the rng state are independent in the worker processes and we might want a different state for each map call
            for sampler in self.samplers.values():
                sampler.rng = np.random.default_rng(self.random_seed + (self.population_size * self.generation_index))

            self.regrowth_sampler.rng = np.random.default_rng(self.random_seed + (self.population_size * self.generation_index))
            self.context_fixer.rng = np.random.default_rng(self.random_seed + (self.population_size * self.generation_index))

            if (save_interval > 0 and ((self.generation_index % save_interval) == 0) and self.generation_index != total_steps) or self.signal_received:
                self.save(suffix=f'gen_{self.generation_index}', log_message=False)

            if self.signal_received:
                logger.info('Received signal to stop evolution, stopping early...')
                break

    def multiple_evolutionary_steps_parallel(self, num_steps: int, should_tqdm: bool = False,
                                             inner_tqdm: bool = False, use_imap: bool = True,
                                             compute_diversity_metrics: bool = False, save_interval: int = 0, start_step: int = 0,
                                             n_workers: int = 8, chunksize: int = 1, maxtasksperchild: typing.Optional[int] = None):

        if hasattr(self, 'population_initialized') and not self.population_initialized:
            raise ValueError('Cannot run multiple evolutionary steps in parallel without initializing the population first')

        logger.debug(f'Launching multiprocessing pool with {n_workers} workers...')
        with mpp.Pool(n_workers, maxtasksperchild=maxtasksperchild) as pool:
            self.multiple_evolutionary_steps(num_steps, pool, chunksize=chunksize,  # type: ignore
                                             should_tqdm=should_tqdm, inner_tqdm=inner_tqdm,
                                             use_imap=use_imap,
                                             compute_diversity_metrics=compute_diversity_metrics,
                                             start_step=start_step,
                                             save_interval=save_interval)

    def signal_handler(self, *args, **kwargs):
        logger.info(f'Caught signal with args: {args} and kwargs: {kwargs}, will terminate when this evolutionary step is done')
        self.signal_received = True

    def _visualize_sample(self, sample: ASTType, top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005, postprocess_sample: bool = True,
                          feature_keywords_to_print: typing.Optional[typing.List[str]] = None):
        if postprocess_sample:
            sample = self.postprocessor(sample)  # type: ignore

        sample_features = self._proposal_to_features(sample)  # type: ignore
        sample_features_tensor = self._features_to_tensor(sample_features)

        if feature_keywords_to_print is not None:
            print('\nFeatures with keywords:')
            for keyword in feature_keywords_to_print:
                keyword_features = [feature for feature, value in sample_features.items() if keyword in feature and value]
                if len(keyword_features) == 0:
                    keyword_features = None

                print(f'"{keyword}": {keyword_features}')

        evaluate_single_game_energy_contributions(
            self.fitness_function, sample_features_tensor, ast_printer.ast_to_string(sample, '\n'), self.feature_names,
            top_k=top_k, display_overall_features=display_overall_features,
            display_game=display_game, min_display_threshold=min_display_threshold,
            )

    def visualize_sample(self, sample_index: int, top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005,
                         postprocess_sample: bool = True, feature_keywords_to_print: typing.Optional[typing.List[str]] = None):
        self._visualize_sample(self.population[sample_index], top_k, display_overall_features, display_game, min_display_threshold, postprocess_sample, feature_keywords_to_print)

    def visualize_top_sample(self, top_index: int, top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005,
                             postprocess_sample: bool = True, feature_keywords_to_print: typing.Optional[typing.List[str]] = None):
        sample_index = np.argsort(self.fitness_values)[-top_index]
        self.visualize_sample(sample_index, top_k, display_overall_features, display_game, min_display_threshold, postprocess_sample, feature_keywords_to_print)


class MAPElitesWeightStrategy(Enum):
    UNIFORM = 0
    FITNESS_RANK = 1
    UCB = 2
    THOMPSON = 3
    # FITNESS_RANK_AND_UCB = 4


class MAPElitesKeyType(Enum):
    INT = 0
    TUPLE = 1


KeyTypeAnnotation = typing.Union[int, typing.Tuple[int, ...]]

class MAPElitesInitializationStrategy(Enum):
    FIXED_SIZE = 0
    ARCHIVE_SIZE = 1
    ARCHIVE_EXEMPLARS = 2
    ARCHIVE_TOP_EXEMPLARS = 3

PARENT_KEY = 'parent_key'


def count_set_bits(n: int) -> int:
    count = 0
    while (n):
        n &= (n-1)
        count+= 1

    return count


class MAPElitesSampler(PopulationBasedSampler):
    archive_cell_first_occupied: typing.Dict[KeyTypeAnnotation, int]
    archive_metrics_history: typing.List[typing.Dict[str, int | float]]
    custom_featurizer: typing.Optional[BehavioralFeaturizer]
    fitness_rank_weights: np.ndarray
    fitness_values: typing.Dict[KeyTypeAnnotation, float]
    generation_size: int
    good_threshold: float
    great_threshold: float
    initialization_strategy: MAPElitesInitializationStrategy
    key_type: MAPElitesKeyType
    map_elites_feature_names: typing.List[str]
    map_elites_feature_names_or_patterns: typing.List[typing.Union[str, re.Pattern]]
    operators: typing.List[typing.Callable]
    operator_weights: np.ndarray
    population: typing.Dict[KeyTypeAnnotation, ASTType]
    previous_sampler_population_seed_path: typing.Optional[str]
    selector: typing.Optional[Selector]
    selector_kwargs: typing.Dict[str, typing.Any]
    update_fitness_rank_weights: bool
    use_crossover: bool = False
    use_cognitive_operators: bool = False
    weight_strategy: MAPElitesWeightStrategy


    def __init__(self,
                 generation_size: int,
                 key_type: MAPElitesKeyType,
                 weight_strategy: MAPElitesWeightStrategy,
                 initialization_strategy: MAPElitesInitializationStrategy,
                 good_threshold: typing.Optional[float] = None, great_threshold: typing.Optional[float] = None,
                 custom_featurizer: typing.Optional[BehavioralFeaturizer] = None,
                 selector_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 previous_sampler_population_seed_path: typing.Optional[str] = None,
                 initial_candidate_pool_size: typing.Optional[int] = None,
                 use_crossover: bool = False,
                 use_cognitive_operators: bool = False,
                 *args, **kwargs):

        self.generation_size = generation_size
        self.key_type = key_type
        self.weight_strategy = weight_strategy
        self.initialization_strategy = initialization_strategy
        self.custom_featurizer = custom_featurizer

        if selector_kwargs is None:
            selector_kwargs = {}

        # selector_kwargs['generation_size'] = generation_size  # no longer passing to do single-sample updating
        self.selector = None
        self.selector_kwargs = selector_kwargs

        if self.weight_strategy == MAPElitesWeightStrategy.UCB:
            self.selector = UCBSelector(**selector_kwargs)
        elif self.weight_strategy == MAPElitesWeightStrategy.THOMPSON:
            self.selector = ThompsonSamplingSelector(**selector_kwargs)

        self.archive_metrics_history = []
        self.previous_sampler_population_seed_path = previous_sampler_population_seed_path
        self.initial_candidate_pool_size = initial_candidate_pool_size
        self.use_crossover = use_crossover
        self.use_cognitive_operators = use_cognitive_operators
        self.update_fitness_rank_weights = True

        self.operators = None  # type: ignore
        self.operator_weights = None  # type: ignore

        super().__init__(*args, **kwargs)

        # if one of these was not provided, try to read them from the fitness function
        if good_threshold is None or great_threshold is None:
            if not hasattr(self.fitness_function, 'score_dict'):
                raise ValueError('Must provide good_threshold and great_threshold if fitness_function does not have a score_dict attribute')

            min_real_game_fitness =  -1 * self.fitness_function.score_dict['max']
            median_real_game_fitness = -1 * self.fitness_function.score_dict['median']

            self.good_threshold = min_real_game_fitness
            self.great_threshold = median_real_game_fitness

        else:
            self.good_threshold = good_threshold
            self.great_threshold = great_threshold

        argparse_args = kwargs['args']

        if argparse_args.map_elites_behavioral_features_key is not None:
            feature_set_key = argparse_args.map_elites_behavioral_features_key
            if feature_set_key not in BEHAVIORAL_FEATURE_SETS:
                raise ValueError(f'Unknown behavioral feature set {feature_set_key}, must be one of {list(BEHAVIORAL_FEATURE_SETS.keys())}')

            features = BEHAVIORAL_FEATURE_SETS[feature_set_key]
            custom_featurizer = None

        else:  # args.map_elites_custom_behavioral_features_key is not None, or using PCA features
            custom_featurizer = build_behavioral_features_featurizer(
                argparse_args,
                self.grammar_parser,
                self.fitness_featurizer,
                self.feature_names
                )
            features = custom_featurizer.get_feature_names()
            if self.key_type != MAPElitesKeyType.TUPLE:
                logger.info('Setting key type to tuple because custom behavioral features are used')
                self.key_type = MAPElitesKeyType.TUPLE

        self.map_elites_feature_names_or_patterns = features  # type: ignore
        self.custom_featurizer = custom_featurizer

    def _pre_population_sample_setup(self):
        self.map_elites_feature_names = []

        names = set([name for name in self.map_elites_feature_names_or_patterns if isinstance(name, str)])
        patterns = [pattern for pattern in self.map_elites_feature_names_or_patterns if isinstance(pattern, re.Pattern)]

        all_feature_names = self.feature_names if self.custom_featurizer is None else self.custom_featurizer.get_feature_names()

        for feature_name in all_feature_names:
            if feature_name in names:
                self.map_elites_feature_names.append(feature_name)
                names.remove(feature_name)

            else:
                for pattern in patterns:
                    if pattern.match(feature_name):
                        self.map_elites_feature_names.append(feature_name)
                        break

        if len(names) > 0:
            raise ValueError(f'Could not find the following feature names in the list of feature names: {names}')

        logger.info(f'Using the following features for MAP-Elites: {self.map_elites_feature_names}')

        self.population = dict()  #  OrderedDict()
        self.fitness_values = dict()  #  OrderedDict()
        self.archive_cell_first_occupied = dict()  #  OrderedDict()

    def _inner_initialize_population(self):
        '''
        Creates the initial population of the archive by either:
        - randomly sampling from the initial sampler until a specified number of samples are added to the archive (i.e. number of cells are filled)
        - loading a previous population from a file
        '''
        if self.previous_sampler_population_seed_path is None:

            # Create initial population by generating self.population_size random samples
            if self.initialization_strategy == MAPElitesInitializationStrategy.FIXED_SIZE:
                super()._inner_initialize_population()

            # Create initial population by generating random samples until the archive has self.population_size samples
            elif self.initialization_strategy == MAPElitesInitializationStrategy.ARCHIVE_SIZE:

                pbar = tqdm(total=self.population_size, desc="Generating initial population of fixed archive size")  # type: ignore
                current_population_size = 0

                while current_population_size < self.population_size:
                    game = self._gen_init_sample(len(self.population))
                    game_fitness, game_features = self._score_proposal(game, return_features=True)  # type: ignore
                    game_key = self._features_to_key(game, game_features)
                    self._add_to_archive(game, game_fitness, game_key, generation_index=-1)

                    if len(self.population) > current_population_size:
                        pbar.update(1)
                        current_population_size = len(self.population)

            # Create initial population by generating random samples until the archive has at least one sample for each feature value
            elif self.initialization_strategy == MAPElitesInitializationStrategy.ARCHIVE_EXEMPLARS:
                if self.custom_featurizer is not None:
                    n_values_by_feature = self.custom_featurizer.get_feature_value_counts()

                else:
                    n_values_by_feature = {feature_name: 2 for feature_name in self.map_elites_feature_names}

                total_feature_count = sum(n_values_by_feature.values())

                pbar = tqdm(total=total_feature_count, desc="Initial archive")  # type: ignore

                feature_values_in_archive = set()
                current_population_size = 0

                postfix = {}

                num_samples_generated = 0
                while len(feature_values_in_archive) < total_feature_count:
                    game = self._gen_init_sample(len(self.population))
                    game_fitness, game_features = self._score_proposal(game, return_features=True)  # type: ignore

                    if self.custom_featurizer is not None:
                        game_key, game_features = self._features_to_key(game, game_features, return_features=True)  # type: ignore

                    else:
                        game_key = self._features_to_key(game, game_features)

                    self._add_to_archive(game, game_fitness, game_key, generation_index=-1)

                    for feature in self.map_elites_feature_names:
                        feature_values_in_archive.add(f"{feature}_{game_features[feature]}")

                    n_feature_values_in = len(feature_values_in_archive)

                    if n_feature_values_in > current_population_size:
                        pbar.update(n_feature_values_in - current_population_size)
                        num_samples_generated = 0
                        current_population_size = n_feature_values_in

                        postfix = {
                            feature_name if len(feature_name) <= 12 else feature_name[:12] + '...': f'{sum([1 for feature_value in feature_values_in_archive if feature_value.startswith(feature_name)])}/{n_values_by_feature[feature_name]}'
                            for feature_name in self.map_elites_feature_names
                        }

                    num_samples_generated += 1
                    postfix['Current'] = num_samples_generated  # type: ignore
                    pbar.set_postfix(postfix)

                logger.info(f'Generating random samples to fill the rest of the population (current population size: {current_population_size}), required population size: {self.population_size}')
                pbar = tqdm(total=self.population_size - current_population_size, desc="Filling archive to initial population size")  # type: ignore

                while current_population_size < self.population_size:
                    game = self._gen_init_sample(len(self.population))
                    game_fitness, game_features = self._score_proposal(game, return_features=True)  # type: ignore
                    game_key = self._features_to_key(game, game_features)
                    self._add_to_archive(game, game_fitness, game_key, generation_index=-1)

                    if len(self.population) > current_population_size:
                        pbar.update(1)
                        current_population_size = len(self.population)


            # Create initial population by sampling a large number of games, then following the logic above
            elif self.initialization_strategy == MAPElitesInitializationStrategy.ARCHIVE_TOP_EXEMPLARS:
                if self.initial_candidate_pool_size is None:
                    raise ValueError('Must provide initial_candidate_pool_size when using ARCHIVE_TOP_EXEMPLARS initialization strategy')

                initial_candidates = []
                for i in trange(self.initial_candidate_pool_size, desc='Generating initial candidates'):
                    game = self._gen_init_sample(i)
                    game_fitness, game_features = self._score_proposal(game, return_features=True)  # type: ignore
                    initial_candidates.append((game_fitness, game, game_features))

                initial_candidates.sort(key=lambda x: x[0], reverse=True)

                if self.custom_featurizer is not None:
                    n_values_by_feature = self.custom_featurizer.get_feature_value_counts()

                else:
                    n_values_by_feature = {feature_name: 2 for feature_name in self.map_elites_feature_names}

                total_feature_count = sum(n_values_by_feature.values())

                pbar = tqdm(total=total_feature_count, desc="Initial archive")  # type: ignore

                feature_values_in_archive = set()
                current_population_size = 0
                current_index = 0

                postfix = {}

                while len(feature_values_in_archive) < total_feature_count:
                    if current_index >= len(initial_candidates):
                        logger.info(f'Exhausted initial candidates with # featres in archive {current_population_size}, stopping early')
                        break

                    game_fitness, game, game_features = initial_candidates[current_index]
                    current_index += 1

                    if self.custom_featurizer is not None:
                        game_key, game_features = self._features_to_key(game, game_features, return_features=True)  # type: ignore

                    else:
                        game_key = self._features_to_key(game, game_features)

                    self._add_to_archive(game, game_fitness, game_key, generation_index=-1)

                    for feature in self.map_elites_feature_names:
                        feature_values_in_archive.add(f"{feature}_{game_features[feature]}")

                    n_feature_values_in = len(feature_values_in_archive)

                    if n_feature_values_in > current_population_size:
                        pbar.update(n_feature_values_in - current_population_size)
                        current_population_size = n_feature_values_in

                        postfix = {
                            feature_name if len(feature_name) <= 12 else feature_name[:12] + '...': f'{sum([1 for feature_value in feature_values_in_archive if feature_value.startswith(feature_name)])}/{n_values_by_feature[feature_name]}'
                            for feature_name in self.map_elites_feature_names
                            if feature_name in n_values_by_feature
                        }

                    postfix['Current'] = current_index  # type: ignore
                    pbar.set_postfix(postfix)

                current_population_size = len(self.population)
                logger.info(f'Adding random samples to fill the rest of the population (current population size: {current_population_size}), required population size: {self.population_size}')
                pbar = tqdm(total=self.population_size - current_population_size, desc="Filling archive to initial population size")  # type: ignore

                while current_population_size < self.population_size:
                    if current_index >= len(initial_candidates):
                        logger.info(f'Exhausted initial candidates with population size {current_population_size}, stopping early')
                        break

                    game_fitness, game, game_features = initial_candidates[current_index]
                    current_index += 1
                    game_key = self._features_to_key(game, game_features)
                    self._add_to_archive(game, game_fitness, game_key, generation_index=-1)

                    if len(self.population) > current_population_size:
                        pbar.update(1)
                        current_population_size = len(self.population)

            else:
                raise ValueError(f'Invalid initialization strategy: {self.initialization_strategy}')

        else:
            logger.info(f'Loading population from {self.previous_sampler_population_seed_path}')
            previous_map_elites = load_data_from_path(self.previous_sampler_population_seed_path)
            for game in tqdm(previous_map_elites.population.values(), desc='Loading previous population', total=len(previous_map_elites.population)):  # type: ignore
                game_fitness, game_features = self._score_proposal(game, return_features=True)  # type: ignore
                game_key = self._features_to_key(game, game_features)
                self._add_to_archive(game, game_fitness, game_key)

            self._update_population_statistics()
            logger.info(f'Loaded {len(self.population)} games from {self.previous_sampler_population_seed_path} with mean fitness {self.mean_fitness:.2f} and std {self.std_fitness:.2f}')

        feature_value_counters = {feature_name: Counter() for feature_name in self.map_elites_feature_names}
        for key in self.population:
            for i, feature_name in enumerate(self.map_elites_feature_names):
                feature_value_counters[feature_name][self._key_value_at_index(key, i)] += 1

        logger.info(f'Initialized population with {len(self.population)} games in the archive with mean fitness {np.mean(list(self.fitness_values.values())):.4f}')
        feature_value_mesage = '\n'.join(['With the following feature-value occupancy:'] + [f'\t{feature_name}: {feature_value_counters[feature_name]}' for feature_name in self.map_elites_feature_names])
        logger.info(feature_value_mesage)

    def _features_to_key(self, game: ASTType, features: typing.Dict[str, float], return_features: bool = False) -> typing.Union[None, int, typing.Tuple[int]]:
        try:
            if self.custom_featurizer is not None:
                features = self.custom_featurizer.get_game_features(game, features)

            if self.key_type.name == MAPElitesKeyType.INT.name:
                key =  sum([(2 ** i) * int(features[feature_name])
                    for i, feature_name in enumerate(self.map_elites_feature_names)
                ])

            elif self.key_type.name == MAPElitesKeyType.TUPLE.name:
                key = tuple([int(features[feature_name]) for feature_name in self.map_elites_feature_names])

            else:
                raise ValueError(f'Unknown key type {self.key_type}')

            if return_features:
                return key, features  # type: ignore

            return key  # type: ignore

        except SamplingException:
            return None

    def _build_evolutionary_step_param_iterator(self, parent_iterator: typing.Iterable[typing.Union[ASTType, typing.List[ASTType]]]) -> typing.Iterable[typing.Tuple[typing.Union[ASTType, typing.List[ASTType]], int, int]]:
        '''
        Given an iterator over parents, return an iterator over tuples of (parent, generation_index, sample_index, return_features)
        '''

        return zip(parent_iterator, itertools.repeat(self.generation_index), itertools.count(), itertools.repeat(True))  # type: ignore

    def _update_population_statistics(self):
        self.population_size = len(self.population)
        fitness_values = list(self.fitness_values.values())
        self.best_fitness = max(fitness_values)
        self.mean_fitness = np.mean(fitness_values)
        self.std_fitness = np.std(fitness_values)

    def set_population(self, population: typing.List[Any], fitness_values: typing.List[float] | None = None):
        keys = None
        features = None
        if fitness_values is None:
            fitness_values, features = zip(*[self._score_proposal(game, return_features=True) for game in population])   # type: ignore

        if features is None:
            keys = [self._features_to_key(game, self._proposal_to_features(game)) for game in population]

        else:
            keys = [self._features_to_key(game, feature) for (game, feature) in zip(population, features)]

        for sample, fitness, key in zip(population, fitness_values, keys):  # type: ignore
            self._add_to_archive(sample, fitness, key)

    def _add_to_archive(self, candidate: ASTType, fitness_value: float, key: typing.Optional[KeyTypeAnnotation], parent_info: typing.Optional[typing.Dict[str, typing.Any]] = None,
                        generation_index: typing.Optional[int] = None, operator: typing.Optional[str] = None):
        '''
        Determines whether a provided candidate should be added to the archive. By default, this happens if the candidate is in a previously unoccupied
        cell or if the candidate has a higher fitness than the candidate already in the cell. If the candidate is added to the archive, the fitness rank
        of each cell is updated. If a selector is provided, the selector is also updated with the parent information (i.e. the cell that produced the candidate)
        '''
        successful = False

        if key is None:
            return

        if key not in self.population:
            successful = True
            generation_index = generation_index if generation_index is not None else self.generation_index
            self.archive_cell_first_occupied[key] = generation_index + 1

        else:
            successful = fitness_value >= self.fitness_values[key]

        if successful:
            self.population[key] = candidate
            self.fitness_values[key] = fitness_value
            self.update_fitness_rank_weights = True
            if operator is not None:
                self.success_by_generation_and_operator[-1][operator] += 1

        if self.selector is not None and parent_info is not None:
            self.selector.update(parent_info[PARENT_KEY], int(successful))

    def _handle_single_operator_results(self, results: SingleStepResults):
        try:
            for candidate, fitness_value, features, parent_info, operator in zip(results.samples, results.fitness_scores, results.sample_features, results.parent_infos, results.operators):
                key = self._features_to_key(candidate, features)
                self._add_to_archive(candidate, fitness_value, key, parent_info, operator=operator)   # type: ignore
        except Exception as e:
            logger.error(f'Exception caught in _handle_single_operator_results: {e}')
            logger.error(traceback.format_exc())
            raise e

    def _end_single_evolutionary_step(self, samples: typing.Optional[SingleStepResults] = None):
        self._update_population_statistics()

    def _custom_tqdm_postfix(self):
        metrics = {
            '# Cells': self.population_size,
            '# Good': len([True for fitness in self.fitness_values.values() if fitness > self.good_threshold]),
            '# Great': len([True for fitness in self.fitness_values.values() if fitness > self.great_threshold]),
            'Timestamp': datetime.now().strftime('%H:%M:%S'),
        }
        self.archive_metrics_history.append(metrics)  # type: ignore
        return metrics

    def _get_parent_iterator(self, n_parents_per_sample: int = 1, n_times_each_parent: int = 1):
        try:
            keys = list(self.population.keys())

            weights = None
            if self.weight_strategy == MAPElitesWeightStrategy.UNIFORM:
                pass

            elif self.weight_strategy == MAPElitesWeightStrategy.FITNESS_RANK:
                if self.update_fitness_rank_weights or len(keys) != len(self.fitness_rank_weights):
                    fitness_values = np.array(list(self.fitness_values.values()))
                    n = len(fitness_values)
                    ranks = n - np.argsort(fitness_values)
                    self.fitness_rank_weights = 0.5 + (ranks / n)
                    self.fitness_rank_weights /= self.fitness_rank_weights.sum()  # type: ignore
                    self.update_fitness_rank_weights = False

                weights = self.fitness_rank_weights

            for _ in range(self.generation_size):
                if self.use_crossover:
                    next_keys = self._get_next_key(keys, weights, n=2)  # type: ignore
                    yield ([self.population[next_keys[0]], self.population[next_keys[1]]], [{PARENT_KEY: next_keys[0]}, {PARENT_KEY: next_keys[1]}])  # type: ignore

                else:
                    key = self._get_next_key(keys, weights)  # type: ignore
                    yield (self.population[key], {PARENT_KEY: key})

        except RuntimeError as e:
            logger.error(f'RuntimeError caught in _get_parent_iterator: {e}')
            logger.error(traceback.format_exc())
            raise e

        except Exception as e:
            logger.error(f'Exception caught in _get_parent_iterator: {e}')
            logger.error(traceback.format_exc())
            raise e

    def _get_next_key(self, keys: typing.List[KeyTypeAnnotation], weights: np.ndarray, n: int = 1) -> KeyTypeAnnotation:
        if self.selector is None:
            key = self._choice(keys, weights=weights, n=n)  # type: ignore
        else:
            key = self.selector.select(keys, rng=self.rng, n=n)
        return key  #  type: ignore

    def _get_operator(self, rng):
        if self.operators is None:
            basic_operators = [self._gen_regrowth_sample, self._insert, self._delete]

            if self.use_crossover:
                basic_operators.extend([self._crossover, self._crossover_insert])  # type: ignore

            if not self.use_cognitive_operators:
                self.operators = basic_operators
                self.operator_weights = np.ones(len(self.operators)) / len(self.operators)

            else:
                cognitive_operators = [self._resample_variable_types, self._resample_first_condition, self._resample_last_condition,
                                    self._resample_variable_types_and_first_condition, self._resample_variable_types_and_last_condition]
                section_resample_operators = [self._sample_or_resample_setup, self._sample_or_resample_terminal, self._resample_scoring, self._crossover_full_sections]

                basic_operator_weights = [0.5 / len(basic_operators)] * len(basic_operators)
                cognitive_operator_weights = [0.3 / len(cognitive_operators)] * len(cognitive_operators)
                section_resample_operator_weights = [0.2 / len(section_resample_operators)] * len(section_resample_operators)

                self.operators = basic_operators + cognitive_operators + section_resample_operators
                self.operator_weights = np.array(basic_operator_weights + cognitive_operator_weights + section_resample_operator_weights)
                self.operator_weights /= self.operator_weights.sum()

        return self._choice(self.operators, rng=rng, weights=self.operator_weights)  # type: ignore

    def _key_to_feature_dict(self, key: KeyTypeAnnotation):
        if self.key_type.name == MAPElitesKeyType.INT.name:
            return {f: (key >> i) % 2 for i, f in enumerate(self.map_elites_feature_names)}  # type: ignore

        elif self.key_type.name == MAPElitesKeyType.TUPLE.name:
            return {f: key[i] for i, f in enumerate(self.map_elites_feature_names)}  # type: ignore

        else:
            raise ValueError(f'Unknown key type {self.key_type}')

    def print_key_features(self, key: KeyTypeAnnotation):
        key_dict = self._key_to_feature_dict(key)
        print(f'Sample features for key {key}:')
        for feature_name, feature_value in key_dict.items():
            print(f'{feature_name}: {feature_value}')

    def _visualize_sample_by_key(self, key: KeyTypeAnnotation, top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005,
                                 postprocess_sample: bool = True, feature_keywords_to_print: typing.Optional[typing.List[str]] = None):
        if key not in self.population:
            raise ValueError(f'Key {key} not found in population')

        self.print_key_features(key)

        self._visualize_sample(self.population[key], top_k, display_overall_features, display_game, min_display_threshold, postprocess_sample, feature_keywords_to_print)
        return key

    def visualize_sample(self, sample_index: int, top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005,
                         postprocess_sample: bool = True, feature_keywords_to_print: typing.Optional[typing.List[str]] = None):
        population_keys = list(self.population.keys())
        return self._visualize_sample_by_key(population_keys[sample_index], top_k, display_overall_features, display_game, min_display_threshold, postprocess_sample, feature_keywords_to_print)

    def visualize_random_sample(self, sample_index: int, top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005,
                         postprocess_sample: bool = True, feature_keywords_to_print: typing.Optional[typing.List[str]] = None):

        population_keys = list(self.population.keys())
        key = self._choice(population_keys, rng=self.rng)
        return self._visualize_sample_by_key(key, top_k, display_overall_features, display_game, min_display_threshold, postprocess_sample, feature_keywords_to_print)

    def _key_value_at_index(self, key: KeyTypeAnnotation, index: int):
        if isinstance(key, int):
            return (key >> index) % 2
        else:
            return key[index]

    def top_sample_key(self, top_index: int, features: typing.Optional[typing.Dict[str, int]] = None, n_features_on: typing.Optional[int] = None):
        if top_index < 1:
            top_index = 1

        if features is not None:
            if any(f not in self.map_elites_feature_names for f in features.keys()):
                raise ValueError(f'Feature names ({list(features.keys())}) must be in {self.map_elites_feature_names}')

            keys = list(self.population.keys())
            feature_to_index = {f: i for i, f in enumerate(self.map_elites_feature_names)}
            relevant_keys = [key for key in keys if all(feature_value == self._key_value_at_index(key, feature_to_index[feature_name]) for feature_name, feature_value in features.items())]
            if len(relevant_keys) == 0:
                print(f'No samples found with features {features}')
                return

        else:
            relevant_keys = self.fitness_values.keys()

        if n_features_on is not None:
            if self.key_type.name == MAPElitesKeyType.INT.name:
                relevant_keys = [key for key in relevant_keys if count_set_bits(key) == n_features_on]  # type: ignore
            elif self.key_type.name == MAPElitesKeyType.TUPLE.name:
                relevant_keys = [key for key in relevant_keys if sum(k != 0 for k in key) == n_features_on]  # type: ignore
            else:
                raise ValueError(f'Unknown key type {self.key_type}')

        if len(relevant_keys) == 0:
            print(f'No samples found with features {features} and {n_features_on} features on')
            return

        fitness_values_and_keys = [(self.fitness_values[key], key) for key in relevant_keys]
        fitness_values_and_keys.sort(key=lambda x: x[0])
        return fitness_values_and_keys[-top_index][1]

    def visualize_top_sample(self, top_index: int, top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005,
                             postprocess_sample: bool = True, feature_keywords_to_print: typing.Optional[typing.List[str]] = None,
                             features: typing.Optional[typing.Dict[str, int]] = None, n_features_on: typing.Optional[int] = None):

        key = self.top_sample_key(top_index, features, n_features_on)
        if key is None:
            return

        return self._visualize_sample_by_key(key, top_k, display_overall_features, display_game, min_display_threshold, postprocess_sample, feature_keywords_to_print)

    def _best_individual(self):
        fitness_values_and_keys = [(fitness, key) for key, fitness in self.fitness_values.items()]
        fitness_values_and_keys.sort(key=lambda x: x[0])
        key = fitness_values_and_keys[-1][1]
        return self.population[key]


feature_names = [f'length_of_then_modals_{i}' for i in range(3, 6)]
def filter_samples_then_three_or_longer(sample: ASTType, sample_features: typing.Dict[str, int], sample_fitness: float) -> bool:
    return any(sample_features[name] for name in feature_names) or bool(sample_features['at_end_found'])


def filter_samples_no_identical_preferences(sample: ASTType, sample_features: typing.Dict[str, int], sample_fitness: float) -> bool:
    preferences = [section for section in sample if isinstance(section, tuple) and section[0] == '(:constraints'][0][1]
    pref_defs = preferences.preferences
    if not isinstance(pref_defs, list):
        return True

    pref_bodies = []
    for pref_def in pref_defs:
        pref = pref_def.definition
        # We don't apply this logic currently to inner preferenecs inside a forall -- we could if that proves to be a mistake
        if pref.parseinfo.rule == 'pref_forall':
            pref_bodies.append(pref)

        else:  # pref.parseinfo.rule == 'preferences'
            pref_body = pref.pref_body.body
            if pref_body.parseinfo.rule == 'pref_body_exists':
                pref_body = pref_body.exists_args

            pref_bodies.append(pref_body)

    pref_body_strs = typing.cast(typing.List[str], [ast_printer.ast_section_to_string(pref, ast_parser.PREFERENCES) for pref in pref_bodies])
    cleaned_strs = set()

    for s in pref_body_strs:
        preference_variables = set(VARIABLE_PATTERN.findall(s))
        for v in preference_variables:
            s = s.replace(v, 'var')

        pref_index = s.find('(preference')
        if pref_index == -1:
            cleaned_strs.add(s.strip())

        else:
            space_index = s.index(' ', pref_index)
            space_after_pref_name_index = s.index(' ', space_index + 1)
            cleaned_strs.add(s[space_after_pref_name_index:].strip())

    return len(pref_defs) == len(cleaned_strs)


SAMPLE_FILTER_FUNCS = {
    'then_three_or_longer': filter_samples_then_three_or_longer,
    'no_identical_preferences': filter_samples_no_identical_preferences,
}


BEHAVIORAL_FEATURE_SETS = {
            'compositionality_structures': [
                re.compile(r'compositionality_structure_.*'),
            ],
            'compositionality_structures_num_preferences_sections': [
                re.compile(r'compositionality_structure_.*'),
                'section_doesnt_exist_setup',
                'section_doesnt_exist_terminal',
                'num_preferences_defined_1',
                'num_preferences_defined_2',
                'num_preferences_defined_3',
            ],
            'compositionality_num_prefs_setup_smaller': [
                # re.compile(r'compositionality_structure_.*'),
                re.compile(r'compositionality_structure_[01234]'),
                'section_doesnt_exist_setup',
                # 'section_doesnt_exist_terminal',
                'num_preferences_defined_1',
                'num_preferences_defined_2',
                # 'num_preferences_defined_3',
            ],
            'mixture_1': [
                # re.compile(r'compositionality_structure_.*'),
                'at_end_found',
                'length_of_then_modals_2',
                'length_of_then_modals_3',
                'section_doesnt_exist_setup',
                'section_doesnt_exist_terminal',
                'num_preferences_defined_1',
                'num_preferences_defined_2',
                'num_preferences_defined_3',
                'in_motion_arg_types_balls_constraints',
                'on_arg_types_blocks_blocks_constraints'
            ],
            'filter_func_experiment_features': [
                'length_of_then_modals_3',
                'section_doesnt_exist_setup',
                'section_doesnt_exist_terminal',
                'num_preferences_defined_1',
                'num_preferences_defined_2',
                'num_preferences_defined_3',
                'in_motion_arg_types_balls_constraints',
                'on_arg_types_blocks_blocks_constraints',
                'in_arg_types_receptacles_balls_constraints',
                'adjacent_arg_types_agent_room_features_constraints',
                'agent_holds_arg_types_blocks_constraints',
            ],
            'length_and_depth_features': [
                # re.compile(r'max_depth_[\w\d_]+'),
                re.compile(r'mean_depth_[\w\d_]+'),
                # re.compile(r'node_count_[\w\d_]+'),
                re.compile(r'num_preferences_defined_[123]'),
            ]
        }


def main(args):
    logger.info('Starting MAP-Elites sampler by cleaning up duckdb temp folder')
    hostname = platform.node().split('.', 1)[0]
    host_duckdb_tmp_folder = os.path.join(DUCKDB_TMP_FOLDER, hostname)

    if os.path.exists(host_duckdb_tmp_folder):
        shutil.rmtree(host_duckdb_tmp_folder)

    if os.path.exists(DUCKDB_QUERY_LOG_FOLDER):
        shutil.rmtree(DUCKDB_QUERY_LOG_FOLDER)

    os.makedirs(host_duckdb_tmp_folder, exist_ok=True)
    os.makedirs(DUCKDB_QUERY_LOG_FOLDER, exist_ok=True)

    if args.use_specific_objects_models:
        raise ValueError('Specific objects models are not currently supported')
        # if args.fitness_function_date_id == DEFAULT_FITNESS_FUNCTION_DATE_ID:
        #     logger.info(f'Using specific objects fitness function date id LATEST_SPECIFIC_OBJECTS_FITNESS_FUNCTION_DATE_ID="{LATEST_SPECIFIC_OBJECTS_FITNESS_FUNCTION_DATE_ID}"')
        #     args.fitness_function_date_id = LATEST_SPECIFIC_OBJECTS_FITNESS_FUNCTION_DATE_ID

        # if args.fitness_featurizer_path == DEFAULT_FITNESS_FEATURIZER_PATH:
        #     logger.info(f'Using specific objects fitness featurizer path LATEST_SPECIFIC_OBJECTS_FITNESS_FEATURIZER_PATH="{LATEST_SPECIFIC_OBJECTS_FITNESS_FEATURIZER_PATH}"')
        #     args.fitness_featurizer_path = LATEST_SPECIFIC_OBJECTS_FITNESS_FEATURIZER_PATH

        # if args.ngram_model_path == LATEST_AST_N_GRAM_MODEL_PATH:
        #     logger.info(f'Using specific objects ngram model path LATEST_SPECIFIC_OBJECTS_AST_N_GRAM_MODEL_PATH="{LATEST_SPECIFIC_OBJECTS_AST_N_GRAM_MODEL_PATH}"')
        #     args.ngram_model_path = LATEST_SPECIFIC_OBJECTS_AST_N_GRAM_MODEL_PATH


    sampler_kwargs = dict(
        omit_rules=args.omit_rules,
        omit_tokens=args.omit_tokens,
    )

    if len(args.sampler_prior_count) == 0:
        logging.info(f'No prior count specified, using default of {PRIOR_COUNT}')
        args.sampler_prior_count = [PRIOR_COUNT]

    vars_args = vars(args)
    vars_args['sample_filter_func'] = None
    if args.sampler_filter_func_key is not None:
        if args.sampler_filter_func_key not in SAMPLE_FILTER_FUNCS:
            raise ValueError(f'Unknown sample filter function {args.sampler_filter_func_key}, must be one of {list(SAMPLE_FILTER_FUNCS.keys())}')

        vars_args['sample_filter_func'] = SAMPLE_FILTER_FUNCS[args.sampler_filter_func_key]

    vars_args['commit_hash'] = Repo(search_parent_directories=True).head.object.hexsha

    if args.sampler_type == MAP_ELITES:
        if args.map_elites_use_crossover:
            logger.info('Using crossover in MAP-Elites => emitting two samples at a time => halving generation and chunk sizes')
            args.map_elites_generation_size = args.map_elites_generation_size // 2
            args.parallel_chunksize = args.parallel_chunksize // 2
            logger.info(f'New generation size: {args.map_elites_generation_size}, new chunksize: {args.parallel_chunksize}')

        if args.map_elites_custom_behavioral_features_key is not None:
            if not args.output_name.endswith(args.map_elites_custom_behavioral_features_key):
                args.output_name += f'_{args.map_elites_custom_behavioral_features_key}'

        elif args.map_elites_pca_behavioral_features_indices is not None:
            args.output_name += f'_pca_{"_".join([str(i) for i in args.map_elites_pca_behavioral_features_indices])}'

        args.map_elites_weight_strategy = MAPElitesWeightStrategy(args.map_elites_weight_strategy)
        weight_strategy_name = args.map_elites_weight_strategy.name.lower()
        if not args.output_name.endswith(weight_strategy_name):
            args.output_name += f'_{weight_strategy_name}'

        args.output_name += f'_seed_{args.random_seed}'

        logger.info(f'Final output name: {args.output_name}')
        evosampler = None

        if args.resume:
            logger.info('Trying to resume from previous run')
            output_name = args.output_name + '_gen_*'
            relevant_files = []
            load_path = None
            latest_generation = None

            for days_back in range(args.resume_max_days_back + 1):
                output_glob_path = get_data_path(args.output_folder, output_name, args.relative_path, delta=timedelta(days=days_back))
                relevant_files = glob.glob(output_glob_path)
                if len(relevant_files) > 0:
                    logger.info(f'Found {len(relevant_files)} relevant files {days_back} days back from {output_glob_path}')
                    break

            if len(relevant_files) == 0:
                logger.info(f'No relevant files found up to {args.resume_max_days_back} days back, starting from scratch')
                args.resume = False

            else:
                gen_index = relevant_files[0].find('gen_')
                number_start_index = gen_index + 4
                generation_to_path = {int(path[number_start_index:path.find('_', number_start_index)]): path for path in relevant_files}
                latest_generation = max(generation_to_path.keys())
                load_path = generation_to_path[latest_generation]

            if load_path is not None:
                logger.info(f'Loading sampler with latest generation {latest_generation} from {load_path}')
                evosampler = typing.cast(MAPElitesSampler, load_data_from_path(load_path))
                logger.info(f'Loaded sample with saving set to {evosampler.saving}')
                evosampler.saving = False
                args.start_step = int(latest_generation)  # type: ignore

        if evosampler is None:
            evosampler = MAPElitesSampler(
                key_type=MAPElitesKeyType(args.map_elites_key_type),
                generation_size=args.map_elites_generation_size,
                weight_strategy=args.map_elites_weight_strategy,
                initialization_strategy=MAPElitesInitializationStrategy(args.map_elites_initialization_strategy),
                good_threshold=args.map_elites_good_threshold,
                great_threshold=args.map_elites_great_threshold,
                previous_sampler_population_seed_path=args.map_elites_population_seed_path,
                initial_candidate_pool_size=args.map_elites_initial_candidate_pool_size,
                use_crossover=args.map_elites_use_crossover,
                use_cognitive_operators=args.map_elites_use_cognitive_operators,
                args=args,
                population_size=args.population_size,
                verbose=args.verbose,
                initial_proposal_type=InitialProposalSamplerType(args.initial_proposal_type),
                fitness_featurizer_path=args.fitness_featurizer_path,
                fitness_function_date_id=args.fitness_function_date_id,
                fitness_function_model_name=args.fitness_function_model_name,
                flip_fitness_sign=not args.no_flip_fitness_sign,
                ngram_model_path=args.ngram_model_path,
                sampler_kwargs=sampler_kwargs,
                relative_path=args.relative_path,
                output_folder=args.output_folder,
                output_name=args.output_name,
                sample_parallel=args.sample_parallel,
                n_workers=args.parallel_n_workers,
                sample_filter_func=args.sample_filter_func,
                sampler_prior_count=args.sampler_prior_count,
                weight_insert_delete_nodes_by_length=not args.no_weight_insert_delete_nodes_by_length,
                sample_patience=args.sample_patience,
                max_sample_depth=args.max_sample_depth,
                max_sample_nodes=args.max_sample_nodes,
                max_sample_total_size=args.max_sample_total_size,
            )

            evosampler.initialize_population()

    else:
        raise ValueError(f'Unknown sampler type {args.sampler_type}')

    if args.wandb:
        wandb.init(
            name=args.output_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            dir=os.environ.get('WANDB_CACHE_DIR', '.wandb'),
        )

    signal.signal(signal.SIGTERM, evosampler.signal_handler)
    signal.signal(signal.SIGUSR1, evosampler.signal_handler)
    signal.signal(signal.SIGUSR2, evosampler.signal_handler)

    tracer = None

    try:
        if args.profile:
            tracer = VizTracer()
            tracer.start()

        if args.sample_parallel:
            evosampler.multiple_evolutionary_steps_parallel(
                num_steps=args.n_steps, should_tqdm=args.should_tqdm, inner_tqdm=args.within_step_tqdm,
                use_imap=not args.parallel_use_plain_map,
                n_workers=args.parallel_n_workers, chunksize=args.parallel_chunksize,
                maxtasksperchild=args.parallel_maxtasksperchild,
                compute_diversity_metrics=args.compute_diversity_metrics,
                save_interval=args.save_interval, start_step=args.start_step
                )

        else:
            evosampler.multiple_evolutionary_steps(
                total_steps=args.n_steps, should_tqdm=args.should_tqdm,
                inner_tqdm=args.within_step_tqdm,
                compute_diversity_metrics=args.compute_diversity_metrics,
                save_interval=args.save_interval,
                )

        # print('Best individual:')
        # evosampler._print_game(evosampler._best_individual())

    except Exception as e:
        exception_caught = True
        logger.error(f'Exception caught in main while trying to sample: {e}')
        logger.error(traceback.format_exc())

    except:
        exception_caught = True
        logger.exception('Unknown exception caught')

    else:
        exception_caught = False

    finally:
        evosampler.save(suffix=f'gen_{evosampler.generation_index}_final' if not exception_caught else 'error')
        if os.path.exists(host_duckdb_tmp_folder):
            shutil.rmtree(host_duckdb_tmp_folder)

        if os.path.exists(DUCKDB_QUERY_LOG_FOLDER):
            shutil.rmtree(DUCKDB_QUERY_LOG_FOLDER)

        if tracer is not None:
            tracer.stop()
            profile_output_path = os.path.join(args.profile_output_folder, args.profile_output_file)
            logger.info(f'Saving profile to {profile_output_path}')
            tracer.save(profile_output_path)

        if args.wandb:
            wandb.finish()


if __name__ == '__main__':
    # torch.set_num_threads(1)

    args = parser.parse_args()
    multiprocessing.set_start_method(args.parallel_start_method, force=True)

    args.grammar_file = os.path.join(args.relative_path, args.grammar_file)
    args.counter_output_path = os.path.join(args.relative_path, args.counter_output_path)
    args.fitness_featurizer_path = os.path.join(args.relative_path, args.fitness_featurizer_path)
    args.ngram_model_path = os.path.join(args.relative_path, args.ngram_model_path)

    args_str = '\n'.join([f'{" " * 26}{k}: {v}' for k, v in vars(args).items()])
    logger.debug(f'Shell arguments:\n{args_str}')

    main(args)
