import cachetools
import cProfile
import duckdb
from functools import reduce
import glob
import gzip
import hashlib
import heapq
import io
from itertools import chain, groupby, permutations, product, repeat, starmap
import json
import logging
import os
import operator
import pandas as pd
import pathlib
import pickle
import polars as pl
pl.enable_string_cache(True)
import pstats
import tatsu, tatsu.ast, tatsu.grammars
import time
from tqdm import tqdm
import typing
from viztracer import VizTracer


from config import COLORS, META_TYPES, TYPES_TO_META_TYPE, OBJECTS_BY_ROOM_AND_TYPE, ORIENTATIONS, SIDES, UNITY_PSEUDO_OBJECTS, NAMED_WALLS, SPECIFIC_NAMED_OBJECTS_BY_ROOM, OBJECT_ID_TO_SPECIFIC_NAME_BY_ROOM, GAME_OBJECT, GAME_OBJECT_EXCLUDED_TYPES
from utils import (extract_predicate_function_name,
                   extract_variables,
                   extract_variable_type_mapping,
                   get_project_dir,
                   get_object_assignments,
                   FullState,
                   AgentState)
from ast_parser import PREFERENCES
from ast_printer import ast_section_to_string
from building_handler import BuildingHandler
from predicate_handler import PREDICATE_LIBRARY_RAW


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.getLogger('wandb').setLevel(logging.WARNING)


COMMON_SENSE_PREDICATES_AND_FUNCTIONS = (
    ("above", 2),
    ("adjacent", 2),
    ("agent_crouches", 0),
    ("agent_holds", 1),
    ("broken", 1),
    ("equal_x_position", 2),
    ("equal_z_position", 2),
    ("in", 2),
    ("in_motion", 1),
    ("object_orientation", 1),  # as it takes 1 object and an orientation we'll hard-code
    ("on", 2),
    ("open", 1),
    ("toggled_on", 1),
    ("touch", 2),
    # ("between", 3),
)

INTERVALS_LIST_POLARS_TYPE = pl.List(pl.List(pl.Int64))


# Maps from types returned by unity to the types used in the DSL
TYPE_REMAP = {
    "alarmclock": "alarm_clock",
    "bridgeblock": "bridge_block",
    "creditcard": "credit_card",
    "cubeblock": "cube_block",
    "curvedramp": "curved_wooden_ramp",
    "cylinderblock": "cylindrical_block",
    "desklamp": "lamp",
    "dogbed": "doggie_bed",
    "flatrectblock": "flat_block",
    "garbagecan": "hexagonal_bin",
    "keychain": "key_chain",
    "longcylinderblock": "tall_cylindrical_block",
    "lightswitch": "main_light_switch",
    "pyramidblock": "pyramid_block",
    "sidetable": "side_table",
    "smallslide": "triangular_ramp",
    "tallrectblock": "tall_rectangular_block",
    "teddybear": "teddy_bear",
    "triangleblock": "triangle_block"
}

DEBUG = False
PROFILE = True
DEFAULT_CACHE_DIR = pathlib.Path(get_project_dir() + '/reward-machine/caches')
DEFAULT_CACHE_FILE_NAME_FORMAT = 'predicate_statistics_with_string_intervals_{traces_hash}.pkl.gz'
NO_INTERVALS_CACHE_FILE_NAME_FORMAT = 'predicate_statistics_no_intervals_{traces_hash}.pkl.gz'
DEFAULT_TRACE_LENGTHS_FILE_NAME_FORMAT = 'trace_lengths_{traces_hash}.pkl'
DEFAULT_IN_PROCESS_TRACES_FILE_NAME_FORMAT = 'in_progress_traces_{traces_hash}.pkl'
DEFAULT_BASE_TRACE_PATH = os.path.join(os.path.dirname(__file__), "traces/participant-traces/")


DEFAULT_COLUMNS = ['predicate', 'arg_1_id', 'arg_1_type', 'arg_2_id', 'arg_2_type', 'trace_id', 'domain', 'intervals']
FULL_PARTICIPANT_TRACE_SET = [os.path.splitext(os.path.basename(t))[0] for t in  glob.glob(os.path.join(DEFAULT_BASE_TRACE_PATH, '*.json'))]


class PredicateNotImplementedException(Exception):
    pass


class MissingVariableException(Exception):
    pass


def stable_hash(str_data: str):
    return hashlib.md5(bytearray(str_data, 'utf-8')).hexdigest()


def stable_hash_list(list_data: typing.Sequence[str]):
    return stable_hash('\n'.join(sorted(list_data)))


TEMP_TABLE_PREFIX = "temp_table_"
DEBUG = False
MAX_CACHE_SIZE = 256
MAX_N_TABLES = 512

if MAX_N_TABLES <= MAX_CACHE_SIZE:
    raise ValueError(f"MAX_N_TABLES ({MAX_N_TABLES}) must be greater than MAX_CACHE_SIZE ({MAX_CACHE_SIZE})")


class LRUCacheWithCallback(cachetools.LRUCache):
    def __init__(self, maxsize: int, evict_callback: typing.Callable[[typing.Any, typing.Any], None], *args, **kwargs):
        super().__init__(maxsize, *args, **kwargs)
        self.evict_callback = evict_callback

    def popitem(self):
        key, value = super().popitem()
        self.evict_callback(key, value)
        return key, value


class MaxRowsLRUCache(cachetools.LRUCache):
    def __init__(self, maxsize: int, max_rows: int, *args, **kwargs):
        super().__init__(maxsize, *args, **kwargs)
        self.max_rows = max_rows

    def __setitem__(self, key, value):
        if isinstance(value, list):
            length = len(value)

        else:
            length = value[0].shape[0]

        if length > self.max_rows:
            # logger.info(f'Rejecting cache entry of type {type(value)} because of length {length}1')
            raise ValueError('Too many rows to cache')

        return super().__setitem__(key, value)

    def popitem(self):
        key, value = super().popitem()
        del value
        return key, None


class CommonSensePredicateStatisticsDatabse():
    data: pd.DataFrame
    domains: typing.List[str]
    predicates: typing.List[str]
    room_objects_cache: typing.Dict[str, typing.Set[str]]
    # same_type_arg_cache: typing.Dict[str, typing.List[typing.Tuple[str, str, str, str]]]
    trace_lengths_and_domains: typing.Dict[str, typing.Tuple[int, str]]
    trace_lengths_and_domains_df: pl.DataFrame

    def __init__(self,
                 use_no_intervals: bool = True,
                 cache_dir: typing.Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
                 trace_names: typing.Sequence[str] = FULL_PARTICIPANT_TRACE_SET,
                 base_trace_path: typing.Union[str, pathlib.Path] = DEFAULT_BASE_TRACE_PATH,
                 cache_filename_format: str = DEFAULT_CACHE_FILE_NAME_FORMAT,
                 no_intervals_cache_filename_format: str = NO_INTERVALS_CACHE_FILE_NAME_FORMAT,
                 trace_lengths_filename_format: str = DEFAULT_TRACE_LENGTHS_FILE_NAME_FORMAT,
                 in_progress_traces_filename_format: str = DEFAULT_IN_PROCESS_TRACES_FILE_NAME_FORMAT,
                 force_trace_names_hash: typing.Optional[str] = None,
                 temp_table_prefix: str = TEMP_TABLE_PREFIX,
                 max_n_tables: int = MAX_N_TABLES,
                 overwrite: bool = False, trace_hash_n_characters: int = 8):

        self.available_table_indices = list(range(max_n_tables))
        heapq.heapify(self.available_table_indices)

        self.cache_dir = cache_dir
        self.temp_table_prefix = temp_table_prefix

        self.object_assignment_cache = {}
        self.room_objects_cache = {}
        # self.temp_table_index = 0

        self.cache = cachetools.LRUCache(maxsize=MAX_CACHE_SIZE)  # , evict_callback=self._cache_evict_callback)
        # self.cache = LRUCacheWithCallback(maxsize=MAX_CACHE_SIZE, evict_callback=self._cache_evict_callback)
        self.object_assignment_cache = MaxRowsLRUCache(maxsize=MAX_CACHE_SIZE, max_rows=MAX_CACHE_SIZE)

        # Compute hash of trace names
        if force_trace_names_hash is not None:
            # logger.info(f"Forcing trace names hash to {force_trace_names_hash}")
            trace_names_hash = force_trace_names_hash
        else:
            trace_names_hash = stable_hash_list([os.path.basename(trace_name).lower().replace(".json", "") for trace_name in trace_names])[:trace_hash_n_characters]

        filename_format_to_use = no_intervals_cache_filename_format if use_no_intervals else cache_filename_format
        self.stats_filename = os.path.join(cache_dir, filename_format_to_use.format(traces_hash=trace_names_hash))
        self.trace_lengths_and_domains_filename = os.path.join(cache_dir, trace_lengths_filename_format.format(traces_hash=trace_names_hash))
        self.in_progress_traces_filename = os.path.join(cache_dir, in_progress_traces_filename_format.format(traces_hash=trace_names_hash))

        self._init_data_and_database(overwrite=overwrite)

    def __getstate__(self) -> typing.Dict[str, typing.Any]:
        state = self.__dict__.copy()
        if 'data' in state: del state['data']
        if 'trace_lengths_and_domains_df' in state: del state['trace_lengths_and_domains_df']
        return state

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        self.__dict__.update(state)
        self._init_data_and_database(False)

    def _init_data_and_database(self, overwrite: bool = False):
        open_method = gzip.open if self.stats_filename.endswith('.gz') else open

        if not overwrite:
            if os.path.exists(self.trace_lengths_and_domains_filename):
                with open_method(self.trace_lengths_and_domains_filename, 'rb') as f:
                    self.trace_lengths_and_domains = pickle.load(f)
            else:
                raise ValueError(f"Trace lengths and domains file {self.trace_lengths_and_domains_filename} does not exist")

        else:
            raise NotImplemented("TODO: Implement this")

            # if force_trace_names_hash is not None:
            #     raise ValueError("Must not specify force_trace_names_hash if cache file does not exist")

            # if base_trace_path is None:
            #     raise ValueError("Must specify base_trace_path if cache file does not exist")

            # logger.info(f"No cache file found at {stats_filename}, building from scratch...")

            # trace_paths = list(sorted([os.path.join(base_trace_path, f"{trace_name}.json" if not trace_name.lower().endswith(".json") else trace_name) for trace_name in trace_names]))

            # if os.path.exists(in_progress_traces_filename):
            #     with open(in_progress_traces_filename, 'rb') as f:
            #         in_progress_trace_paths = pickle.load(f)

            # else:
            #     in_progress_trace_paths = []

            # if len(in_progress_trace_paths) == len(trace_paths):
            #     logger.info(f"Foud as many in progres traces as there are total, so starting over")
            #     in_progress_trace_paths = []

            # if len(in_progress_trace_paths) > 0:
            #     trace_paths = [trace_path for trace_path in trace_paths if trace_path not in in_progress_trace_paths]
            #     logger.info(f"Found {len(in_progress_trace_paths)} in progress traces, resuming processing with {len(trace_paths)} traces remaining")
            #     self.data = pd.read_pickle(stats_filename)  # type: ignore
            #     logger.info(f'Loaded data with shape {self.data.shape} from {stats_filename}')
            #     with open_method(trace_lengths_and_domains_filename, 'rb') as f:
            #         self.trace_lengths_and_domains = pickle.load(f)

            # else:
            #     logger.info(f"No in progress traces found, starting from scratch")
            #     # TODO (gd1279): if we ever decide to support 3- or 4- argument predicates, we'll need to
            #     # add additional columns here
            #     self.data = pd.DataFrame(columns=DEFAULT_COLUMNS)  # type: ignore
            #     self.trace_lengths_and_domains = {}

            # for trace_path in tqdm(trace_paths, desc="Processing traces"):
            #     trace = json.load(open(trace_path, 'r'))
            #     self.process_trace(trace)
            #     self.data.to_pickle(stats_filename)

            #     with open_method(trace_lengths_and_domains_filename, 'wb') as f:
            #         pickle.dump(self.trace_lengths_and_domains, f)

            #     in_progress_trace_paths.append(trace_path)
            #     with open(in_progress_traces_filename, 'wb') as f:
            #         pickle.dump(in_progress_trace_paths, f)

            # # if we've reached the end, remove the in-progress file
            # os.remove(in_progress_traces_filename)

            # # TODO: convert all interval lists to strings

        self._trace_lengths_and_domains_to_df()

        self.max_length = self.trace_lengths_and_domains_df.select(pl.col('trace_length').max()).item()

        self._create_databases()

    # def _cache_evict_callback(self, cache_key, cache_value):
    #     # logger.info(f"Evicting {cache_key} => {cache_value} from cache")
    #     table_name, _ = cache_value
    #     duckdb.sql(f"DROP TABLE {table_name}")
    #     table_index = int(table_name.replace(self.temp_table_prefix, ''))
    #     heapq.heappush(self.available_table_indices, table_index)

    def _create_databases(self):
        table_query = duckdb.sql("SHOW TABLES")
        if table_query is not None:
            all_tables = set(t.lower() for t in chain.from_iterable(table_query.fetchall()))
            if 'data' in all_tables:
                # logger.info('Skipping creating tables because they already exist')
                self.domains = set(chain.from_iterable(duckdb.sql("SELECT unnest(enum_range(NULL::domain))").fetchall()))  # type: ignore
                self.predicates = set(chain.from_iterable(duckdb.sql("SELECT unnest(enum_range(NULL::predicate))").fetchall()))  # type: ignore
                return

        logger.info('Loading data before creating database')
        if os.path.exists(self.stats_filename):
            self.data = pd.read_pickle(self.stats_filename)
        else:
            raise ValueError(f"Stats file {self.stats_filename} does not exist")

        self.domains = list(self.data['domain'].unique())  # type: ignore
        self.predicates = list(self.data['predicate'].unique())  # type: ignore

        logger.info("Creating DuckDB table...")

        # TODO: restore the string_intervals if we restore intervals-based logic
        data_df = self.data
        drop_columns = [c for c in ('intervals', 'string_intervals') if c in data_df.columns]
        if len(drop_columns) > 0:
            data_df = data_df.drop(columns=drop_columns)

        all_predicates = tuple(data_df.predicate.unique())
        duckdb.sql(f"CREATE TYPE predicate AS ENUM {all_predicates};")
        all_domains = tuple(data_df.domain.unique())
        duckdb.sql(f"CREATE TYPE domain AS ENUM {all_domains};")
        all_trace_ids = tuple(data_df.trace_id.unique())
        duckdb.sql(f"CREATE TYPE trace_id AS ENUM {all_trace_ids};")
        all_types = tuple([t for t in set(data_df.arg_1_type.unique()) | set(data_df.arg_2_type.unique()) if isinstance(t, str) ])
        duckdb.sql(f"CREATE TYPE arg_type AS ENUM {all_types};")

        all_ids = set(data_df.arg_1_id.unique()) | set(data_df.arg_2_id.unique()) | set(UNITY_PSEUDO_OBJECTS.keys())
        for room_types in OBJECTS_BY_ROOM_AND_TYPE.values():
            for object_types in room_types.values():
                all_ids.update(object_types)

        all_ids = tuple([t for t in all_ids if isinstance(t, str)])
        duckdb.sql(f"CREATE TYPE arg_id AS ENUM {all_ids};")
        logger.info("Done creating enums, about to create table")

        duckdb.sql("CREATE TABLE data(predicate predicate, arg_1_id arg_id, arg_1_type arg_type, arg_2_id arg_id, arg_2_type arg_type, trace_id trace_id, domain domain);")

        duckdb.sql("INSERT INTO data SELECT * FROM data_df")
        data_rows = duckdb.sql("SELECT count(*) FROM data").fetchone()[0]  # type: ignore
        logger.info(f"Loaded data, found {data_rows} rows")

        duckdb.create_function("empty_bitstring", self.create_empty_bitstring_function(self.max_length), [], duckdb.typing.BIT)  # type: ignore

        del self.data

    def _trace_lengths_and_domains_to_df(self):
        trace_ids = []
        trace_lengths = []
        domains = []

        for trace_id, (length, domain) in self.trace_lengths_and_domains.items():
            trace_ids.append(trace_id)
            trace_lengths.append(length)
            domains.append(domain)

        self.trace_lengths_and_domains_df = pl.DataFrame(dict(
            trace_id=trace_ids,
            trace_length=trace_lengths,
            domain=domains,
        ), schema_overrides=dict(
            trace_id=pl.Categorical,
            domain=pl.Categorical,
        ))

    def _get_room_objects(self, trace) -> typing.Set[str]:
        '''
        Returns the set of objects in the room type of the given trace, excluding pseudo-objects,
        colors, and the agent
        '''
        room_type = self._domain_key(trace['scene'])
        if room_type not in self.room_objects_cache:
            room_objects = set(sum([list(OBJECTS_BY_ROOM_AND_TYPE[room_type][obj_type]) for obj_type in OBJECTS_BY_ROOM_AND_TYPE[room_type]], []))
            room_objects -= set(COLORS)
            room_objects -= set(SIDES)
            room_objects -= set(ORIENTATIONS)
            room_objects.remove('agent')
            self.room_objects_cache[room_type] = room_objects

        return self.room_objects_cache[room_type]

    def process_trace(self, trace):
        '''
        Process a trace, collecting the intervals in which each predicate is true (for
        every possible set of its arguments). Adds the information to the overall dataframe
        '''

        room_objects = self._get_room_objects(trace)
        replay = trace['replay']
        replay_len = int(len(replay))

        # Maps from the predicate-arg key to a list of intervals in which the predicate is true
        predicate_satisfaction_mapping = {}

        # Stores the most recent state of the agent and of each object
        most_recent_agent_state = None
        most_recent_object_states = {}
        initial_object_states = {}

        received_full_update = False

        full_trace_id = f"{trace['id']}-{trace['replayKey']}"
        domain_key = self._domain_key(trace['scene'])
        building_handler = BuildingHandler(domain_key)

        for idx, state in tqdm(enumerate(replay), total=replay_len, desc=f"Processing replay {full_trace_id}", leave=False):
            is_final = idx == replay_len - 1
            state = FullState.from_state_dict(state)

            building_handler.process(state)

            # Track changes to the agent
            if state.agent_state_changed:
                most_recent_agent_state = state.agent_state

            # And to objects
            objects_with_initial_rotations = []
            for obj in state.objects:
                if obj.object_id not in initial_object_states:
                    initial_object_states[obj.object_id] = obj

                obj = obj._replace(initial_rotation=initial_object_states[obj.object_id].rotation)
                objects_with_initial_rotations.append(obj)
                most_recent_object_states[obj.object_id] = obj

            state = state._replace(objects=objects_with_initial_rotations)

            # Check if we've received a full state update, which we detect by seeing if the most_recent_object_states
            # includes every object in the room (aside from PseudoObjects, which never receive updates)
            if not received_full_update:
                difference = room_objects.difference(set(most_recent_object_states.keys()))
                received_full_update = (difference == set(UNITY_PSEUDO_OBJECTS.keys()))

            # Only perform predicate checks if we've received at least one full state update
            if received_full_update and (state.n_objects_changed > 0 or state.agent_state_changed):
                for predicate, n_args in COMMON_SENSE_PREDICATES_AND_FUNCTIONS:

                    # Some predicates take only an empty list for arguments
                    if n_args == 0:
                        possible_args = [[]]

                    # Collect all possible sets of arguments in which at least one has been updated this step
                    else:
                        changed_this_step = [obj.object_id for obj in state.objects]

                        if state.agent_state_changed:
                            changed_this_step.append("agent")

                        room_objects_with_active_buildings_only = room_objects - (building_handler.building_id_set - building_handler.active_buildings)
                        possible_args = list(product(*([changed_this_step] + list(repeat(room_objects_with_active_buildings_only, n_args - 1)))))

                        # Filter out any sets of arguments with duplicates
                        possible_args = [arg_set for arg_set in possible_args if len(set(arg_set)) == len(arg_set)]

                    for arg_set in possible_args:
                        arg_assignments = permutations(arg_set) if predicate != 'object_orientation' else [(arg_set[0], orientation) for orientation in ORIENTATIONS]
                        # TODO: similar things for colors and side, if that ever comes back

                        for arg_ids in arg_assignments:

                            args, arg_types = [], []
                            for obj_id in arg_ids:
                                if obj_id == "agent":
                                    args.append(most_recent_agent_state)
                                    arg_types.append("agent")

                                # This will assign each of the specific walls (e.g. 'north_wall') to object type 'wall',
                                # which is correct, but we also need to assign each of them to the type which is their
                                # name in order to account for cases where something like 'north_wall' is used directly
                                elif obj_id in UNITY_PSEUDO_OBJECTS:
                                    args.append(UNITY_PSEUDO_OBJECTS[obj_id])
                                    arg_types.append(UNITY_PSEUDO_OBJECTS[obj_id].object_type.lower())

                                elif (obj_id in ORIENTATIONS) or (obj_id in COLORS) or (obj_id in SIDES):
                                    args.append(obj_id)
                                    arg_types.append(obj_id)

                                else:
                                    args.append(most_recent_object_states[obj_id])
                                    arg_types.append(most_recent_object_states[obj_id].object_type.lower())

                            key = self._predicate_key(predicate, arg_ids)
                            predicate_fn = PREDICATE_LIBRARY_RAW[predicate]

                            evaluation = predicate_fn(most_recent_agent_state, args)

                            # If the predicate is true, then check to see if the last interval is closed. If it is, then
                            # create a new interval
                            if evaluation:
                                if key not in predicate_satisfaction_mapping:
                                    info = {"predicate": predicate,"trace_id": full_trace_id,
                                            "domain": domain_key,
                                            "intervals": [[idx, None]]}

                                    for i, (arg_id, arg_type) in enumerate(zip(arg_ids, arg_types)):
                                        info[f"arg_{i + 1}_id"] = arg_id
                                        info[f"arg_{i + 1}_type"] = TYPE_REMAP.get(arg_type, arg_type)
                                    predicate_satisfaction_mapping[key] = info

                                elif predicate_satisfaction_mapping[key]['intervals'][-1][1] is not None:
                                    predicate_satisfaction_mapping[key]['intervals'].append([idx, None])

                            # If the predicate is false, then check to see if the last interval is open. If it is, then
                            # close it
                            else:
                                if key in predicate_satisfaction_mapping and predicate_satisfaction_mapping[key]["intervals"][-1][1] is None:
                                    predicate_satisfaction_mapping[key]["intervals"][-1][1] = idx


        # Close any intervals that are still open
        for key in predicate_satisfaction_mapping:
            if predicate_satisfaction_mapping[key]["intervals"][-1][1] is None:
                predicate_satisfaction_mapping[key]["intervals"][-1][1] = replay_len

        # Record the trace's length
        self.trace_lengths_and_domains[full_trace_id] = (replay_len, domain_key)

        # Collapse the intervals into a single dataframe
        game_df = pd.DataFrame(predicate_satisfaction_mapping.values())

        # Extract the rows in which an argument is one of the specific named objects
        object_ids_to_specific_names = {id: name for name, ids in SPECIFIC_NAMED_OBJECTS_BY_ROOM[domain_key].items() for id in ids}
        object_ids_to_specific_names.update({id: id for id in NAMED_WALLS})
        specific_objects = list(object_ids_to_specific_names.keys())

        # Remap the arg ids and types for the specific objects
        sub_df = game_df.loc[(game_df["arg_1_id"].isin(specific_objects)) | (game_df["arg_2_id"].isin(specific_objects))].copy()
        specific_object_new_rows = []
        for idx, row in sub_df.iterrows():
            first_arg_specific = row["arg_1_id"] in object_ids_to_specific_names
            second_arg_specific = row["arg_2_id"] in object_ids_to_specific_names

            # If both are specific, we need to create three copies of the row --
            # one each with the first arg, second arg, and both args replaced with the specific names
            if first_arg_specific and second_arg_specific:
                row_dict = row.to_dict()
                first_arg_copy = row_dict.copy()
                second_arg_copy = row_dict.copy()
                first_arg_copy["arg_1_type"] = object_ids_to_specific_names[row["arg_1_id"]]
                second_arg_copy["arg_2_type"] = object_ids_to_specific_names[row["arg_2_id"]]
                specific_object_new_rows.append(first_arg_copy)
                specific_object_new_rows.append(second_arg_copy)
                # fall through to the other cases to create the row with both copied

            if first_arg_specific:
                sub_df.at[idx, "arg_1_type"] = object_ids_to_specific_names[row["arg_1_id"]]
                # sub_df.at[idx, "arg_1_id"] = row["arg_1_type"]

            if second_arg_specific:
                sub_df.at[idx, "arg_2_type"] = object_ids_to_specific_names[row["arg_2_id"]]
                # sub_df.at[idx, "arg_2_id"] = sub_df.at[idx, "arg_2_type"]

        specific_objects_new_rows_df = pd.DataFrame(specific_object_new_rows)

        # Combine the resulting dataframes and add them to the overall dataframe
        self.data = pd.concat([self.data, game_df, specific_objects_new_rows_df, sub_df], ignore_index=True)  # type: ignore

    def _predicate_key(self, predicate: str, args: typing.Sequence[str]) -> str:
        return f"{predicate}-({','.join(args)})"

    def _domain_key(self, domain: str):
        if domain.endswith('few_new_objects'):
            return 'few'
        elif domain.endswith('semi_sparse_new_objects'):
            return 'medium'
        elif domain.endswith('many_new_objects'):
            return 'many'
        else:
            raise ValueError(f"Unrecognized domain: {domain}")

    def _object_assignments_cache_key(self, domain, variable_types, used_objects = None):
        '''
        Returns a key for the object assignments cache
        '''
        return (domain, tuple(variable_types), tuple(used_objects) if used_objects is not None else None)

    @cachetools.cachedmethod(cache=operator.attrgetter('object_assignment_cache'), key=_object_assignments_cache_key)
    def _object_assignments(self, domain, variable_types, used_objects = None):
        '''
        Wrapper around get_object_assignments in order to cache outputs
        '''
        return get_object_assignments(domain, variable_types, used_objects=used_objects)


    def filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs):
        # temp_table_names = [t[0] for t in duckdb.sql('SHOW TABLES').fetchall() if t[0].startswith(self.temp_table_prefix)]
        # for table_name in temp_table_names:
        #     duckdb.sql(f'DROP TABLE {table_name};')

        try:
            # self.temp_table_index = 0
            result, _ = self._inner_filter(predicate, mapping, **kwargs)
            return result
            # print(outcome_table_name, self.cache.currsize)
            # n_traces = duckdb.sql(f"SELECT COUNT(DISTINCT(trace_id)) FROM {outcome_table_name}").fetchone()[0]  # type: ignore
            # n_intervals = duckdb.sql(f"SELECT COUNT(*) FROM {result}").fetchone()[0]  # type: ignore
            # n_total_states = duckdb.sql(f"SELECT SUM(bit_count(string_intervals)) FROM {outcome_table_name}").fetchone()[0] # type: ignore
            # return n_intervals # , n_intervals, n_total_states

        except PredicateNotImplementedException as e:
            # Pass the exception through and let the caller handle it
            raise e

    def _table_name(self, index: int):
        return f"{self.temp_table_prefix}{index}"

    def _next_temp_table_index(self):
        if len(self.available_table_indices) == 0:
            raise RuntimeError("Ran out of available table indices. This shouldn't happen...")

        return heapq.heappop(self.available_table_indices)

    def _next_temp_table_name(self):
        return self._table_name(self._next_temp_table_index())

    def create_empty_bitstring_function(self, length: int):
        def empty_bitstring():
            return '0' * length

        return empty_bitstring

    def _predicate_and_mapping_cache_key(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], *args, **kwargs) -> str:
        '''
        Returns a string that uniquely identifies the predicate and mapping
        '''
        return ast_section_to_string(predicate, PREFERENCES) + "_" + str(mapping)

    @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_predicate(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], return_trace_ids: bool = False, **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        predicate_name = extract_predicate_function_name(predicate)  # type: ignore

        if predicate_name not in self.predicates:
            raise PredicateNotImplementedException(predicate_name)

        variables = extract_variables(predicate)  # type: ignore
        used_variables = set(variables)

        # Restrict the mapping to just the referenced variables and expand meta-types
        relevant_arg_mapping = {}
        for var in variables:
            if var in mapping:
                relevant_arg_mapping[var] = sum([META_TYPES.get(arg_type, [arg_type]) for arg_type in mapping[var]], [])

            # This handles variables which are referenced directly, like the desk and bed
            elif not var.startswith("?"):
                relevant_arg_mapping[var] = [var]

            else:
                raise MissingVariableException(f"Variable {var} is not in the mapping")

        select_items = ["trace_id", "domain", "string_intervals"]
        where_items = [f"predicate='{predicate_name}'"]

        for i, (arg_var, arg_types) in enumerate(relevant_arg_mapping.items()):
            # if it can be the generic object type, we filter for it specifically
            if GAME_OBJECT in arg_types:
                where_items.append(f"arg_{i + 1}_type NOT IN {tuple(GAME_OBJECT_EXCLUDED_TYPES)}")

            else:
                if len(arg_types) == 1:
                    where_items.append(f"arg_{i + 1}_type='{arg_types[0]}'")
                else:
                    where_items.append(f"arg_{i + 1}_type IN {tuple(arg_types)}")

            select_items.append(f"arg_{i + 1}_id as '{arg_var}'")

        select_items = 'DISTINCT(trace_id)' if return_trace_ids else 'COUNT(*)'
        query = f"SELECT {select_items} FROM data WHERE {' AND '.join(where_items)};"

        if return_trace_ids:
            return tuple(chain.from_iterable(duckdb.sql(query).fetchall())), None  # type: ignore

        else:
            return duckdb.sql(query).fetchone()[0], None  # type: ignore

        # table_name = self._next_temp_table_name()
        # query = f"CREATE TEMP TABLE {table_name} AS SELECT {', '.join(select_items)} FROM data WHERE {' AND '.join(where_items)};"
        # if DEBUG: print(query)
        # duckdb.sql(query)
        # return table_name, used_variables

    @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_and(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        and_args = predicate["and_args"]
        if not isinstance(and_args, list):
            and_args = [and_args]

        table_names = []
        used_variables_by_child = []
        all_used_variables = set()

        for and_arg in and_args:  # type: ignore
            try:
                sub_table_name, sub_used_variables = self._inner_filter(and_arg, mapping)  # type: ignore
                table_names.append(sub_table_name)
                used_variables_by_child.append(sub_used_variables)
                all_used_variables |= sub_used_variables

            except PredicateNotImplementedException as e:
                continue

        if len(table_names) == 0:
            raise PredicateNotImplementedException("All sub-predicates of the and were not implemented")

        if len(table_names) == 1:
            return table_names[0], used_variables_by_child[0]

        select_items = [f"{table_names[0]}.trace_id", f"{table_names[0]}.domain"]
        selected_variables = set()
        string_intervals = []
        join_clauses = []

        for i, (sub_table_name, sub_used_variables) in enumerate(zip(table_names, used_variables_by_child)):
            string_intervals.append(f"{sub_table_name}.string_intervals")

            for variable in sub_used_variables:
                if variable not in selected_variables:
                    select_items.append(f'{sub_table_name}."{variable}"')
                    selected_variables.add(variable)

            if i > 0:
                join_parts = [f"INNER JOIN {sub_table_name} ON {table_names[0]}.trace_id={sub_table_name}.trace_id"]

                for j, (prev_table_name, prev_used_variables) in enumerate(zip(table_names[:i], used_variables_by_child[:i])):
                    shared_variables = sub_used_variables & prev_used_variables
                    join_parts.extend([f'{sub_table_name}."{v}"={prev_table_name}."{v}"' for v in shared_variables])

                join_clauses.append(" AND ".join(join_parts))


        select_items.append(f'({" & ".join(string_intervals)}) AS string_intervals')

        table_name = self._next_temp_table_name()
        query = f"CREATE TABLE {table_name} AS SELECT {', '.join(select_items)} FROM {table_names[0]} {' '.join(join_clauses)};"
        if DEBUG: print(query)
        duckdb.sql(query)

        self._cleanup_empty_assignments(table_name)

        return table_name, all_used_variables

    def _cleanup_empty_assignments(self, table_name: str) -> None:
        cleanup_query = f"DELETE FROM {table_name} WHERE bit_count(string_intervals) = 0;"
        if DEBUG: print(cleanup_query)
        duckdb.sql(cleanup_query)

    @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_or(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        or_args = predicate["or_args"]
        if not isinstance(or_args, list):
            or_args = [or_args]

        table_names = []
        used_variables_by_child = []
        all_used_variables = set()

        for or_arg in or_args:  # type: ignore
            try:
                sub_table_name, sub_used_variables = self._inner_filter(or_arg, mapping)  # type: ignore
                table_names.append(sub_table_name)
                used_variables_by_child.append(sub_used_variables)
                all_used_variables |= sub_used_variables

            except PredicateNotImplementedException as e:
                continue

        if len(table_names) == 0:
            raise PredicateNotImplementedException("All sub-predicates of the por were not implemented")

        if len(table_names) == 1:
            return table_names[0], used_variables_by_child[0]

        # Building this table to explicitly represent all potential assignments, instead of implicitly representing them as nulls
        empty_intervals_table_name = self._build_potential_missing_values_table(mapping, list(all_used_variables))

        table_names.insert(0, empty_intervals_table_name)
        used_variables_by_child.insert(0, all_used_variables)

        select_items = [f"{table_names[0]}.trace_id", f"{table_names[0]}.domain"]
        selected_variables = set()
        string_intervals = []
        join_clauses = []

        for i, (sub_table_name, sub_used_variables) in enumerate(zip(table_names, used_variables_by_child)):
            string_intervals.append(f"{sub_table_name}.string_intervals")

            for variable in sub_used_variables:
                if variable not in selected_variables:
                    select_items.append(f'{sub_table_name}."{variable}"')
                    selected_variables.add(variable)

            if i > 0:
                join_parts = [f"LEFT JOIN {sub_table_name} ON {table_names[0]}.trace_id={sub_table_name}.trace_id"]

                shared_variables = sub_used_variables & all_used_variables
                join_parts.extend([f'{sub_table_name}."{v}"={empty_intervals_table_name}."{v}"' for v in shared_variables])

                join_clauses.append(" AND ".join(join_parts))

        string_intervals_coalesce = [f"COALESCE({si}, empty_bitstring())" if i > 0 else si for i, si in enumerate(string_intervals)]
        select_items.append(f'({" | ".join(string_intervals_coalesce)}) AS string_intervals')

        table_name = self._next_temp_table_name()
        query = f"CREATE TABLE {table_name} AS SELECT {', '.join(select_items)} FROM {table_names[0]} {' '.join(join_clauses)};"
        if DEBUG: print(query)
        duckdb.sql(query)

        self._cleanup_empty_assignments(table_name)

        return table_name, all_used_variables


    def _build_potential_missing_values_table(self, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], relevant_vars: typing.List[str]):
        # For each trace ID, and each assignment of the vars that exist in the sub_predicate_df so far:
        relevant_var_mapping = {var: mapping[var] if var.startswith("?") else [var] for var in relevant_vars}
        variable_types = tuple(tuple(relevant_var_mapping[var]) for var in relevant_var_mapping.keys())

        # For each cartesian product of the valid assignments for those vars given the domain
        possible_arg_assignments = [self._object_assignments(domain, variable_types) for domain in self.domains]

        # TODO: if I decide to make the intervals for each trace as long as that trace, I need to revise the logic here to do that, too
        possible_assignments_df = pl.DataFrame(dict(domain=self.domains, assignments=possible_arg_assignments, string_intervals=['0' * self.max_length] * len(self.domains)),
                                                schema=dict(domain=pl.Categorical, assignments=pl.List(pl.List(pl.Categorical)), string_intervals=pl.Utf8))  # type: ignore

        potential_missing_values_df = self.trace_lengths_and_domains_df.join(possible_assignments_df, how="left", on="domain")
        potential_missing_values_df = potential_missing_values_df.explode('assignments').select(
            'domain', 'trace_id',
            pl.col("assignments").list.to_struct(fields=relevant_vars), 'string_intervals').unnest('assignments')

        table_name = self._next_temp_table_name()
        if DEBUG: print(f"Creating potential missing values table {table_name}")

        assignment_columns = ", ".join(f'"{var}" arg_id' for var in relevant_vars)
        duckdb.sql(f"CREATE TABLE {table_name}(domain domain, trace_id trace_id, {assignment_columns}, string_intervals BITSTRING)");
        duckdb.sql(f"INSERT INTO {table_name} SELECT * FROM potential_missing_values_df;")
        # duckdb.sql(f"ALTER TABLE {table_name} ALTER string_intervals TYPE BITSTRING")
        # for var in relevant_vars:
        #     duckdb.sql(f'ALTER TABLE {table_name} ALTER "{var}" TYPE arg_id')

        return table_name

    @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_not(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        try:
            inner_table_name, used_variables = self._inner_filter(predicate["not_args"], mapping)  # type: ignore
        except PredicateNotImplementedException as e:
            raise PredicateNotImplementedException(f"Sub-predicate of the not ({e.args}) was not implemented")


        relevant_vars = list(used_variables)
        potential_missing_values_table_name = self._build_potential_missing_values_table(mapping, relevant_vars)

        # Now, for each possible combination of args on each trace / domain, 'intervals' will contain the truth intervals if
        # they exist and null otherwise, and 'intervals_right' will contain the empty interval'
        join_columns = ["trace_id"] + relevant_vars

        select_items = [f"{potential_missing_values_table_name}.trace_id", f"{potential_missing_values_table_name}.domain"]
        select_items.extend(f"{potential_missing_values_table_name}.\"{var}\"" for var in relevant_vars)
        select_items.append(f"(~( {potential_missing_values_table_name}.string_intervals | COALESCE({inner_table_name}.string_intervals, empty_bitstring()) )) AS string_intervals")

        join_items = [f"{potential_missing_values_table_name}.\"{column}\"={inner_table_name}.\"{column}\""  for column in join_columns]

        table_name = self._next_temp_table_name()
        query = f"CREATE TABLE {table_name} AS SELECT {', '.join(select_items)} FROM {potential_missing_values_table_name} LEFT JOIN {inner_table_name} ON {' AND '.join(join_items)};"
        if DEBUG: print(query)
        duckdb.sql(query)

        self._cleanup_empty_assignments(table_name)

        return table_name, used_variables


    def _inner_filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str]]:
        '''
        Filters the data by the given predicate and mapping, returning a list of intervals in which the predicate is true
        for each processed trace

        Returns a dictionary mapping from the trace ID to a dictionary that maps from the set of arguments to a list of
        intervals in which the predicate is true for that set of arguments
        '''

        predicate_rule = predicate.parseinfo.rule  # type: ignore

        if predicate_rule == "predicate":
            return self._handle_predicate(predicate, mapping, **kwargs)

        elif predicate_rule == "super_predicate":
            return self._inner_filter(predicate["pred"], mapping, **kwargs)  # type: ignore

        elif predicate_rule == "super_predicate_and":
            return self._handle_and(predicate, mapping, **kwargs)

        elif predicate_rule == "super_predicate_or":
            return self._handle_or(predicate, mapping, **kwargs)

        elif predicate_rule == "super_predicate_not":
            return self._handle_not(predicate, mapping, **kwargs)

        elif predicate_rule in ["super_predicate_exists", "super_predicate_forall", "function_comparison"]:
            raise PredicateNotImplementedException(predicate_rule)

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")


if __name__ == '__main__':
    DEFAULT_GRAMMAR_PATH = "./dsl/dsl.ebnf"
    grammar = open(DEFAULT_GRAMMAR_PATH).read()
    grammar_parser = typing.cast(tatsu.grammars.Grammar, tatsu.compile(grammar))

    game = open(get_project_dir() + '/reward-machine/games/ball_to_bin_from_bed.txt').read()
    game_ast = grammar_parser.parse(game)  # type: ignore

    test_pred_orientation = game_ast[3][1]['setup']['and_args'][0]['setup']['exists_args']['setup']['statement']['conserved_pred']['pred']['and_args'][0]['pred']

    # should be: (and (in_motion ?b) (not (agent_holds ?b)))
    test_pred_1 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][1]['seq_func']['hold_pred']

    # should be: (and (not (in_motion ?b)) (in ?h ?b)))
    test_pred_2 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][2]['seq_func']['once_pred']

    # should be: (once (and (not (in_motion ?b) (exists (?c - hexagonal_bin) (in ?c ?b)))))
    # test_pred_3 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][3]['seq_func']['once_pred']

    block_stacking_game = open(get_project_dir() + '/reward-machine/games/block_stacking.txt').read()
    block_stacking_game_ast = grammar_parser.parse(block_stacking_game)  # type: ignore

    test_pred_or = block_stacking_game_ast[3][1]['preferences'][0]['definition']['pref_body']['body']['exists_args']['then_funcs'][2]['seq_func']['hold_pred']
    test_pred_desk_or = block_stacking_game_ast[3][1]['preferences'][1]['definition']['pref_body']['body']['exists_args']['at_end_pred']
    test_pred_agent_as_arg = block_stacking_game_ast[3][1]['preferences'][2]['definition']['pref_body']['body']['exists_args']['at_end_pred']
    # test these with ?g - game_object
    test_pred_object_in_top_drawer = block_stacking_game_ast[3][1]['preferences'][3]['definition']['pref_body']['body']['exists_args']['at_end_pred']
    test_pred_agent_adjacent = block_stacking_game_ast[3][1]['preferences'][4]['definition']['pref_body']['body']['exists_args']['at_end_pred']
    # test with ?s - sliding_door
    test_pred_agent_adjacent = block_stacking_game_ast[3][1]['preferences'][5]['definition']['pref_body']['body']['exists_args']['at_end_pred']

    BALL_TO_BIN_FROM_BED_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/FhhBELRaBFiGGvX0YG7W-preCreateGame-rerecorded.json')
    agent_adj_game = open(get_project_dir() + '/reward-machine/games/test_agent_door_adjacent.txt').read()
    agent_adj_game_ast = grammar_parser.parse(agent_adj_game)  # type: ignore

    agent_adj_predicate = agent_adj_game_ast[3][1]['preferences'][0]['definition']['pref_body']['body']['exists_args']['then_funcs'][0]['seq_func']['once_pred']
    agent_adj_mapping = {"?d": ["ball"]}


    test_mapping = {"?b": ["ball"], "?h": ["hexagonal_bin"]}
    block_test_mapping = {"?b1": ['cube_block'], "?b2": ["cube_block"]}
    block_desk_test_mapping = {"?b": ["block"]}
    all_block_test_mapping = {"?b1": ["block"], "?b2": ["block"]}

    test_predicates_and_mappings = [
        (test_pred_1, test_mapping),
        (test_pred_1, block_desk_test_mapping),
        (test_pred_2, test_mapping),
        (test_pred_or, block_test_mapping),
        (test_pred_or, all_block_test_mapping),
        (test_pred_desk_or, test_mapping),
        (test_pred_desk_or, block_desk_test_mapping),
        (agent_adj_predicate, agent_adj_mapping),
    ]

    stats = CommonSensePredicateStatisticsDatabse(cache_dir=DEFAULT_CACHE_DIR,
                                                    # trace_names=CURRENT_TEST_TRACE_NAMES,
                                                    trace_names=FULL_PARTICIPANT_TRACE_SET,
                                                    # cache_rules=[],
                                                    base_trace_path=DEFAULT_BASE_TRACE_PATH,
                                                    # force_trace_names_hash='028b3733',
                                                    overwrite=True
                                                    )

    variable_type_usage = json.loads(open(f"{get_project_dir()}/reward-machine/caches/variable_type_usage.json", "r").read())
    for var_type in variable_type_usage:
        if var_type in META_TYPES:
            continue

        n_intervals = duckdb.sql(f"SELECT count(*) FROM data WHERE (arg_1_type='{var_type}' OR arg_2_type='{var_type}')").fetchone()[0]  # type: ignore

        prefix = "[+]" if n_intervals > 0 else "[-]"
        print(f"{prefix} {var_type} has {n_intervals} appearances")

    exit()

    # out = stats.filter(test_pred_object_in_top_drawer, {"?g": ["game_object"]})
    # print(out)
    # _print_results_as_expected_intervals(out)

    exit()

    tracer = None
    profile = None
    if PROFILE:
        # tracer = VizTracer(10000000, ignore_c_function=True, ignore_frozen=True)
        # tracer.start()
        profile = cProfile.Profile()
        profile.enable()

    N_ITER = 100
    for i in range(N_ITER):
        # print(f"\n====================")

        for test_pred, test_mapping in test_predicates_and_mappings:
            # print(f"Testing {test_pred} with mapping {test_mapping}")
            out = stats.filter(test_pred, test_mapping)
            # _print_results_as_expected_intervals(out)
        # inner_end = time.perf_counter()
        # print(f"Time per iteration: {'%.5f' % (inner_end - inner_start)}s")

    if profile is not None:
        profile.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    # if tracer is not None:
        # tracer.stop()
        # profile_output_path = os.path.join(get_project_dir(), 'reward-machine/temp/viztracer_split_args.json')
        # print(f'Saving profile to {profile_output_path}')
        # tracer.save(profile_output_path)

    exit()

    # Satisfactions of in_motion
    in_motion_sats = stats.data[stats.data["predicate"] == "in_motion"]

    print("All 'in_motion' satisfactions:")
    print(in_motion_sats)

    # Satisfactions of in
    in_sats = stats.data[stats.data["predicate"] == "in"]
    long_in_sats = in_sats[(in_sats["end_step"] - in_sats["start_step"]) / in_sats["replay_len"] >= 0.9]

    print("All 'in' satisfactions:")
    print(in_sats[["predicate", "arg_ids", "start_step", "end_step", "replay_len"]])

    print("\n\nLong 'in' satisfactions (>90pct of trace):")
    print(long_in_sats[["predicate", "arg_ids", "start_step", "end_step", "replay_len"]])

    # Satisfactions of agent_holds
    print(stats.data[stats.data["predicate"] == "agent_holds"])

    # stats = CommonSensePredicateStatistics(cache_dir, [f"{get_project_dir()}/reward-machine/traces/{trace}-rerecorded.json" for trace in TEST_TRACE_NAMES], overwrite=True)
    # print(stats.data)
