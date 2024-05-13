import cachetools
import cProfile
from functools import reduce
import glob
import gzip
import hashlib
import io
from itertools import groupby, permutations, product, repeat, starmap
import json
import logging
import numpy as np
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


from config import COLORS, META_TYPES, OBJECTS_BY_ROOM_AND_TYPE, ORIENTATIONS, SIDES, UNITY_PSEUDO_OBJECTS, NAMED_WALLS, SPECIFIC_NAMED_OBJECTS_BY_ROOM, GAME_OBJECT, GAME_OBJECT_EXCLUDED_TYPES, SLIDING_DOOR, SHELF_DESK
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


OBJECT_ID_TYPE_REMAP = {
     "Window|+02.28|+00.93|-03.18": SLIDING_DOOR,
     "Window|-01.02|+00.93|-03.19": SLIDING_DOOR,
     "Shelf|+03.13|+00.63|-00.56": SHELF_DESK,
     "Shelf|+03.13|+00.63|-02.27": SHELF_DESK,
}


DEBUG = False
PROFILE = True
DEFAULT_CACHE_DIR = pathlib.Path(get_project_dir() + '/reward-machine/caches')
DEFAULT_CACHE_FILE_NAME_FORMAT = 'predicate_statistics_{traces_hash}.pkl.gz'
BITSTRING_INTERVALS_FILE_NAME_FORMAT = 'predicate_statistics_bitstring_intervals_{traces_hash}.pkl.gz'
DEFAULT_TRACE_LENGTHS_FILE_NAME_FORMAT = 'trace_lengths_{traces_hash}.pkl'
DEFAULT_IN_PROCESS_TRACES_FILE_NAME_FORMAT = 'in_progress_traces_{traces_hash}.pkl'
DEFAULT_BASE_TRACE_PATH = os.path.join(os.path.dirname(__file__), "traces/participant-traces/")


MAX_CACHE_SIZE = 512

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


b = bytes([0, 1])
BYTE_MAPPING = {b: str(i) for i, b in enumerate(b)}


def row_to_string_intervals(row):
    value = np.zeros(row['trace_length'], dtype=np.uint8)
    for interval in row['intervals']:
        value[interval[0]:interval[1]] = 1

    return ''.join(map(lambda b: BYTE_MAPPING[b], value.tobytes()))


def create_bitstings_df(df, trace_lengths_and_domains_dict, output_path):
    trace_lengths_and_domains_rows = [(key, *value) for key, value in trace_lengths_and_domains_dict.items()]
    trace_lengths_and_domains_df = pd.DataFrame(trace_lengths_and_domains_rows, columns=['trace_id', 'trace_length', 'domain'])

    split_args_with_trace_length_df = df.join(trace_lengths_and_domains_df.drop(columns=['domain']).set_index('trace_id'), on='trace_id')
    split_args_with_string_intervals_df = split_args_with_trace_length_df.assign(intervals=split_args_with_trace_length_df.apply(row_to_string_intervals, axis=1))
    print(f'Saving bitstrings df to {output_path}')
    split_args_with_string_intervals_df.to_pickle(output_path)


class CommonSensePredicateStatisticsSplitArgs():
    data: pl.DataFrame
    domains: typing.List[str]
    predicates: typing.List[str]
    room_objects_cache: typing.Dict[str, typing.Set[str]]
    # same_type_arg_cache: typing.Dict[str, typing.List[typing.Tuple[str, str, str, str]]]
    trace_lengths_and_domains: typing.Dict[str, typing.Tuple[int, str]]
    trace_lengths_and_domains_df: pl.DataFrame


    def __init__(self,
                 cache_dir: typing.Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
                 trace_names: typing.Sequence[str] = FULL_PARTICIPANT_TRACE_SET,
                 cache_rules: typing.Optional[typing.Sequence[str]] = None,
                 base_trace_path: typing.Union[str, pathlib.Path] = DEFAULT_BASE_TRACE_PATH,
                 cache_filename_format: str = DEFAULT_CACHE_FILE_NAME_FORMAT,
                 trace_lengths_filename_format: str = DEFAULT_TRACE_LENGTHS_FILE_NAME_FORMAT,
                 in_progress_traces_filename_format: str = DEFAULT_IN_PROCESS_TRACES_FILE_NAME_FORMAT,
                 force_trace_names_hash: typing.Optional[str] = None,
                 overwrite: bool = False, trace_hash_n_characters: int = 8):

        self.cache_dir = cache_dir

        # Cache calls to get_object_assignments
        self.cache_rules = cache_rules
        self.object_assignment_cache = {}
        self.predicate_interval_cache = {}
        self.room_objects_cache = {}
        # self.same_type_arg_cache = {}

        # Compute hash of trace names
        if force_trace_names_hash is not None:
            logger.info(f"Forcing trace names hash to {force_trace_names_hash}")
            self.trace_names_hash = force_trace_names_hash
        else:
            self.trace_names_hash = stable_hash_list([os.path.basename(trace_name).lower().replace(".json", "") for trace_name in trace_names])[:trace_hash_n_characters]

        self.stats_filename = os.path.join(cache_dir, cache_filename_format.format(traces_hash=self.trace_names_hash))
        self.trace_lengths_and_domains_filename = os.path.join(cache_dir, trace_lengths_filename_format.format(traces_hash=self.trace_names_hash))
        in_progress_traces_filename = os.path.join(cache_dir, in_progress_traces_filename_format.format(traces_hash=self.trace_names_hash))
        open_method = gzip.open if self.stats_filename.endswith('.gz') else open

        if os.path.exists(self.stats_filename) and not overwrite:
            self.data = pd.read_pickle(self.stats_filename)  # type: ignore
            logger.info(f'Loaded data with shape {self.data.shape} from {self.stats_filename}')
            with open_method(self.trace_lengths_and_domains_filename, 'rb') as f:
                self.trace_lengths_and_domains = pickle.load(f)

        else:
            if force_trace_names_hash is not None:
                raise ValueError("Must not specify force_trace_names_hash if cache file does not exist")

            if base_trace_path is None:
                raise ValueError("Must specify base_trace_path if cache file does not exist")

            logger.info(f"No cache file found at {self.stats_filename}, building from scratch...")

            trace_paths = list(sorted([os.path.join(base_trace_path, f"{trace_name}.json" if not trace_name.lower().endswith(".json") else trace_name) for trace_name in trace_names]))

            if os.path.exists(in_progress_traces_filename):
                with open(in_progress_traces_filename, 'rb') as f:
                    in_progress_trace_paths = pickle.load(f)

            else:
                in_progress_trace_paths = []

            if len(in_progress_trace_paths) == len(trace_paths):
                logger.info(f"Foud as many in progres traces as there are total, so starting over")
                in_progress_trace_paths = []

            if len(in_progress_trace_paths) > 0:
                trace_paths = [trace_path for trace_path in trace_paths if trace_path not in in_progress_trace_paths]
                logger.info(f"Found {len(in_progress_trace_paths)} in progress traces, resuming processing with {len(trace_paths)} traces remaining")
                self.data = pd.read_pickle(stats_filename)  # type: ignore
                logger.info(f'Loaded data with shape {self.data.shape} from {self.stats_filename}')
                with open_method(self.trace_lengths_and_domains_filename, 'rb') as f:
                    self.trace_lengths_and_domains = pickle.load(f)

            else:
                logger.info(f"No in progress traces found, starting from scratch")
                # TODO (gd1279): if we ever decide to support 3- or 4- argument predicates, we'll need to
                # add additional columns here
                self.data = pd.DataFrame(columns=DEFAULT_COLUMNS)  # type: ignore
                self.trace_lengths_and_domains = {}

            for trace_path in tqdm(trace_paths, desc="Processing traces"):
                trace = json.load(open(trace_path, 'r'))
                self.process_trace(trace)
                self.data.to_pickle(self.stats_filename)

                with open_method(self.trace_lengths_and_domains_filename, 'wb') as f:
                    pickle.dump(self.trace_lengths_and_domains, f)

                in_progress_trace_paths.append(trace_path)
                with open(in_progress_traces_filename, 'wb') as f:
                    pickle.dump(in_progress_trace_paths, f)

            # if we've reached the end, remove the in-progress file
            os.remove(in_progress_traces_filename)

        self.domains = list(self.data['domain'].unique())  # type: ignore
        self.predicates = list(self.data['predicate'].unique())  # type: ignore

        self._trace_lengths_and_domains_to_df()
        self.data = pl.from_pandas(
            self.data, schema_overrides=dict(
                predicate=pl.Categorical,
                trace_id=pl.Categorical,
                domain=pl.Categorical,
                arg_1_id=pl.Categorical,
                arg_1_type=pl.Categorical,
                arg_2_id=pl.Categorical,
                arg_2_type=pl.Categorical
            )
        )

        self.cache = MaxRowsLRUCache(maxsize=MAX_CACHE_SIZE, max_rows=MAX_CACHE_SIZE)
        self.object_assignment_cache = MaxRowsLRUCache(maxsize=MAX_CACHE_SIZE, max_rows=MAX_CACHE_SIZE)

        # Convert to polars
        # self.data = pl.from_pandas(self.data)

        # Removing support for same_type until I regenerate cache -- it's ~80% of the records in the DB
        if 'same_type' in self.predicates:
            self.predicates.remove('same_type')
            self.data = self.data.filter(pl.col('predicate') != 'same_type')
            logger.info(f'After filtering out same_type, data has shape {self.data.shape}')

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

    # def _get_room_same_type_args(self, trace) -> typing.List[typing.Tuple[str, str, str, str]]:
    #     room_type = self._domain_key(trace['scene'])
    #     if room_type not in self.same_type_arg_cache:
    #         same_type_args = []

    #         for obj_type, objects_of_type in OBJECTS_BY_ROOM_AND_TYPE[room_type].items():
    #             meta_type = TYPES_TO_META_TYPE.get(obj_type, None)
    #             for object in objects_of_type:
    #                 same_type_args.append((object, obj_type, obj_type, obj_type))
    #                 if meta_type is not None:
    #                     same_type_args.append((object, obj_type, meta_type, meta_type))
    #                 if object in OBJECT_ID_TO_SPECIFIC_NAME_BY_ROOM[room_type]:
    #                     specific_type = OBJECT_ID_TO_SPECIFIC_NAME_BY_ROOM[room_type][object]
    #                     same_type_args.append((object, obj_type, specific_type, specific_type))

    #             for first_object, second_object in permutations(objects_of_type, 2):
    #                 same_type_args.append((first_object, obj_type, second_object, obj_type))

    #         self.same_type_arg_cache[room_type] = same_type_args

    #     return self.same_type_arg_cache[room_type]

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

    def _intersect_intervals(self, intervals_1: typing.List[typing.List[int]], intervals_2: typing.List[typing.List[int]]):
        '''
        Given two lists of [start, end] intervals, returns the list of intervals in which they intersect

        Code from: https://stackoverflow.com/questions/69997547/intersections-of-intervals
        '''
        intersections = []
        i = j = 0

        while i < len(intervals_1) and j < len(intervals_2):
            low = max(intervals_1[i][0], intervals_2[j][0])
            high = min(intervals_1[i][1], intervals_2[j][1])
            if low <= high:
                intersections.append([low, high])

            # Remove the interval with the smallest endpoint
            if intervals_1[i][1] < intervals_2[j][1]:
                i += 1
            else:
                j += 1

        return intersections


    def _intersect_intervals_tuple(self, intervals: typing.Tuple[typing.List[typing.List[int]], typing.List[typing.List[int]]]):
        if not intervals[0] or not intervals[1]:
            return ([],)

        return (self._intersect_intervals(intervals[0], intervals[1]),)

    def _intersect_many_intervals_tuple(self, intervals: typing.Tuple[typing.List[typing.List[int]], ...]):
        if any(not i for i in intervals):
            return ([],)

        output_intervals = intervals[0]
        for other_intervals in intervals[1:]:
            output_intervals = self._intersect_intervals(output_intervals, other_intervals)
            if not output_intervals: break

        return (output_intervals,)

    def _union_intervals(self, intervals_1: typing.List[typing.List[int]], intervals_2: typing.List[typing.List[int]]):
        '''
        Given two lists of [start, end] intervals, returns the list of intervals in which either is true
        '''
        all_intervals = sorted(intervals_1 + intervals_2)
        unions = []

        for start, end in all_intervals:
            if unions != [] and unions[-1][1] >= start - 1:
                unions[-1][1] = max(unions[-1][1], end)
            else:
                unions.append([start, end])

        return unions

    def _union_intervals_tuple(self, intervals: typing.Tuple[typing.List[typing.List[int]], typing.List[typing.List[int]]]):
        i0, i1 = intervals
        retval = None
        if not i0:
            if not i1:
                retval = []
            else:
                retval = i1
        elif not i1:
            retval = i0

        if retval is None:
            retval = self._union_intervals(intervals[0], intervals[1])

        return (retval, )

    def _union_many_intervals(self, intervals_list: typing.List[typing.List[typing.List[int]]]):
        '''
        Given two lists of [start, end] intervals, returns the list of intervals in which either is true
        '''
        all_intervals = sorted(sum(intervals_list, []))
        unions = []

        for start, end in all_intervals:
            if unions != [] and unions[-1][1] >= start - 1:
                unions[-1][1] = max(unions[-1][1], end)
            else:
                unions.append([start, end])

        return unions

    def _union_many_intervals_tuple(self, intervals: typing.Tuple[typing.List[typing.List[int]], ...]):
        intervals_list = [i for i in intervals if i]
        if len(intervals_list) == 0:
            return ([],)
        elif len(intervals_list) == 1:
            return (intervals_list[0],)

        return (self._union_many_intervals(intervals_list),)

    def _invert_intervals(self, intervals: typing.List[typing.List[int]], length: int):
        if not intervals or intervals[0] is None:
            return [[0, length]]

        inverted = []
        cur = 0

        for interval in intervals:
            if cur < interval[0]:
                inverted.append([cur, interval[0]])
            cur = interval[1]

        if cur < length:
            inverted.append([cur, length])

        return inverted

    def _invert_intervals_tuple_apply(self, intervals_tuple: typing.Tuple[typing.List[typing.List[int]], int]):
        return (self._invert_intervals(*intervals_tuple), )

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
                                        arg_type = TYPE_REMAP.get(arg_type, arg_type)
                                        arg_type = OBJECT_ID_TYPE_REMAP.get(arg_id, arg_type)
                                        info[f"arg_{i + 1}_type"] = arg_type
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

        # Add the same_types intervals
        # same_type_records = [
        #     dict(predicate='same_type', arg_1_id=arg_1_id, arg_1_type=arg_1_type, arg_2_id=arg_2_id, arg_2_type=arg_2_type,
        #          trace_id=full_trace_id, domain=domain_key, intervals=[[0, replay_len]])
        #     for arg_1_id, arg_1_type, arg_2_id, arg_2_type in self._get_room_same_type_args(trace)
        # ]
        # same_type_df = pd.DataFrame(same_type_records)

        # Combine the resulting dataframes and add them to the overall dataframe
        self.data = pd.concat([self.data, game_df, specific_objects_new_rows_df, sub_df], ignore_index=True)  # type: ignore

    def filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]):
        try:
            result, _ = self._inner_filter(predicate, mapping)
            if isinstance(result, int):
                return result

            n_traces = result.select("trace_id").unique().shape[0]
            # if n_traces == 0:
            #     return 0, 0, 0

            return n_traces

            # n_intervals = result.select(pl.col("intervals").list.lengths()).sum().item()
            # if n_intervals == 0:
            #     return 0, 0, 0

            # total_interval_states = result.select("intervals").explode("intervals") \
            #     .select(pl.col("intervals").list.to_struct(fields=["start", "end"])) \
            #     .unnest("intervals").select(pl.col("end") - pl.col("start")).sum().item()
            # # TODO: we could also similarlty extract, mean, or mean per trace, or...
            # return n_traces, n_intervals, total_interval_states
            # sorted_variables = sorted(used_variables)
            # return {(row_dict['trace_id'], tuple([f'{k}->{row_dict[k]}' for k in sorted_variables])): row_dict['intervals']
            #         for row_dict in result.to_dicts()}
        except PredicateNotImplementedException as e:
            # TODO: decide what we return in this case, or if we pass it through and let the feature handle it
            raise e

    def _predicate_and_mapping_cache_key(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], *args, **kwargs) -> str:
        '''
        Returns a string that uniquely identifies the predicate and mapping
        '''
        return ast_section_to_string(predicate, PREFERENCES) + "_" + str(mapping)


    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_predicate(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]) -> typing.Tuple[pl.DataFrame, typing.Set[str]]:
        predicate_name = extract_predicate_function_name(predicate)  # type: ignore

        if predicate_name not in self.predicates:
            raise PredicateNotImplementedException(predicate_name)

        variables = extract_variables(predicate)  # type: ignore
        used_variables = set(variables)

        if DEBUG: start = time.perf_counter()

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

        filter_expr = pl.col("predicate") == predicate_name
        rename_mapping = {}
        drop_columns = ['predicate']
        for i, (arg_var, arg_types) in enumerate(relevant_arg_mapping.items()):
            # if it can be the generic object type, we filter for it specifically
            if GAME_OBJECT in arg_types:
                filter_expr &= pl.col(f"arg_{i + 1}_type").is_in(GAME_OBJECT_EXCLUDED_TYPES).is_not()
            else:
                filter_expr &= pl.col(f"arg_{i + 1}_type").is_in(arg_types)

            # verify that if we query "same_type" and the second argument is a type, we only take the correct rows
            # we then also drop it from the DF as passed forward
            # TODO: figure out if there are any other predicates that require the same behavior
            # if predicate_name == "same_type" and i == 1 and not arg_var.startswith("?"):
            #     filter_expr &= pl.col(f"arg_{i + 1}_id") == arg_var
            #     drop_columns.append(arg_var)
            #     used_variables.remove(arg_var)

            rename_mapping[f"arg_{i + 1}_id"] = arg_var

        # Returns a dataframe in which the arg_id columns are renamed to the variable names they map to
        return self.data.lazy().filter(filter_expr).select(pl.count()).collect().item(), None   # type: ignore
        # predicate_df = self.data.filter(filter_expr).rename(rename_mapping)

        # We drop the arg_type columns and any un-renamed arg_id columns, since they're no longer needed
        # Added a drop of the predicate column which we no longer need here -- discovered it's still here while debugging yesterday
        predicate_df = predicate_df.drop([c for c in predicate_df.columns if c.startswith("arg_")] + drop_columns)

        if DEBUG:
            end = time.perf_counter()
            print(f"Time per collect '{predicate_name}': {'%.5f' % (end - start)}s")  # type: ignore

        return predicate_df, used_variables

    @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_and(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]) -> typing.Tuple[pl.DataFrame, typing.Set[str]]:
        and_args = predicate["and_args"]
        if not isinstance(and_args, list):
            and_args = [and_args]

        sub_predicate_dfs = []
        used_variables = set()
        for and_arg in and_args:  # type: ignore
            try:
                sub_predicate_df, sub_used_variables = self._inner_filter(and_arg, mapping)
                sub_predicate_dfs.append(sub_predicate_df)
                used_variables |= sub_used_variables

            except PredicateNotImplementedException as e:
                continue

        if len(sub_predicate_dfs) == 0:
            raise PredicateNotImplementedException("All sub-predicates of the and were not implemented")

        if len(sub_predicate_dfs) == 1:
            return sub_predicate_dfs[0], used_variables

        predicate_df = sub_predicate_dfs[0]

        suffixes = []
        intervals_columns = ['intervals']
        if DEBUG: start = time.perf_counter()
        for i, sub_predicate_df in enumerate(sub_predicate_dfs[1:]):
            # Collect all variables which appear in both the current predicate (which will be expanded) and the sub-predicate
            shared_var_columns = [c for c in (set(predicate_df.columns) & set(sub_predicate_df.columns) & used_variables)]

            suffix = f"_{i}"
            suffixes.append(suffix)
            intervals_columns.append(f"intervals{suffix}")

            # Join the two dataframes based on the trace identifier, and shared variable columns
            predicate_df = predicate_df.join(sub_predicate_df, how="inner", on=["trace_id"] + shared_var_columns, suffix=suffix)

            if predicate_df.shape[0] == 0:
                predicate_df = predicate_df.drop([c for c in predicate_df.columns if any(c.endswith(suffix) for suffix in suffixes)])
                return predicate_df, used_variables

        # Replace the intervals column with the intersection of all intervals columns
        # predicate_df.replace("intervals", predicate_df.select(*intervals_columns).apply(
        #     self._intersect_many_intervals_tuple, INTERVALS_LIST_POLARS_TYPE)['column_0'])

        # Remove all the 'right-hand' columns added by the joins
        predicate_df = predicate_df.drop([c for c in predicate_df.columns if any(c.endswith(suffix) for suffix in suffixes)])

        # Remove any rows with empty intervals
        # predicate_df = predicate_df.filter(pl.col("intervals").list.lengths() > 0)

        if DEBUG:
            end = time.perf_counter()
            print(f"Time to AND: {'%.5f' % (end - start)}s")  # type: ignore

        return predicate_df, used_variables

    @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_or(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]) -> typing.Tuple[pl.DataFrame, typing.Set[str]]:
        or_args = predicate["or_args"]
        if not isinstance(or_args, list):
            or_args = [or_args]

        sub_predicate_dfs = []
        used_variables = set()

        for or_arg in or_args:  # type: ignore
            try:
                sub_predicate_df, sub_used_variables = self._inner_filter(or_arg, mapping)
                sub_predicate_dfs.append(sub_predicate_df)
                used_variables |= sub_used_variables

            except PredicateNotImplementedException as e:
                continue

        if len(sub_predicate_dfs) == 0:
            raise PredicateNotImplementedException("All sub-predicates of the por were not implemented")

        relevant_vars = list(used_variables)
        # Building this dataframe to explicitly represent all potential assignments, instead of implicitly representing them as nulls
        # as the implicit nulls were making it really hard to write the logic for the (not (or ...)) case
        predicate_df = self._build_potential_missing_values_df(mapping, relevant_vars)

        suffixes = []
        intervals_columns = ['intervals']

        if DEBUG: start = time.perf_counter()
        for i, sub_predicate_df in enumerate(sub_predicate_dfs):
            # Same procedure as with 'and', above, except a union instead of an intersection for the intervals
            shared_var_columns = [c for c in (set(predicate_df.columns) & set(sub_predicate_df.columns) & used_variables)]

            suffix = f"_{i}"
            suffixes.append(suffix)
            intervals_columns.append(f"intervals{suffix}")

            predicate_df = predicate_df.join(sub_predicate_df, how="left", on=["trace_id"] + shared_var_columns, suffix=suffix)

        predicate_df.replace("intervals", predicate_df.select(pl.concat_list(intervals_columns).alias("intervals")).to_series())

        # predicate_df.replace("intervals", predicate_df.select(*intervals_columns).apply(self._union_many_intervals_tuple, INTERVALS_LIST_POLARS_TYPE)['column_0'])
        predicate_df = predicate_df.drop([c for c in predicate_df.columns if any(c.endswith(suffix) for suffix in suffixes)])

        # Only filter out rows with no intervals after joining with all sub-predicates
        # predicate_df = predicate_df.filter(pl.col("intervals").list.lengths() > 0)

        if DEBUG:
            end = time.perf_counter()
            print(f"Time to OR: {'%.5f' % (end - start)}s")  # type: ignore

        return predicate_df, used_variables

    def _build_potential_missing_values_df(self, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], relevant_vars: typing.List[str]):
        if len(relevant_vars) == 0:
            raise MissingVariableException("Attempting to build missing values df with no relevant variables")

        # For each trace ID, and each assignment of the vars that exist in the sub_predicate_df so far:
        relevant_var_mapping = {var: mapping[var] if var.startswith("?") else [var] for var in relevant_vars}
        variable_types = tuple(tuple(relevant_var_mapping[var]) for var in relevant_var_mapping.keys())

        # For each cartesian product of the valid assignments for those vars given the domain
        possible_arg_assignments = [self._object_assignments(domain, variable_types) for domain in self.domains]

        possible_assignments_df = pl.DataFrame(dict(domain=self.domains, assignments=possible_arg_assignments, intervals=[[]] * len(self.domains)),
                                                    schema=dict(domain=pl.Categorical, assignments=pl.List(pl.List(pl.Categorical)), intervals=pl.List(pl.List(pl.Int64))))  # type: ignore

        try:
            if all(len(assignment) == 0 for assignment in possible_arg_assignments):
                raise MissingVariableException("No possible assignments for any variable")

            potential_missing_values_df = self.trace_lengths_and_domains_df.join(possible_assignments_df, how="left", on="domain")
            potential_missing_values_df = potential_missing_values_df.explode('assignments').select(
                'domain', 'trace_id', 'trace_length',
                pl.col("assignments").list.to_struct(fields=relevant_vars), 'intervals').unnest('assignments')

        except Exception as e:
            print("mapping", mapping)
            print("relevant_var_mapping:", relevant_var_mapping)
            print("variable_types:", variable_types)
            print("possible_arg_assignments:", possible_arg_assignments)
            print("assignments", possible_assignments_df.select('assignments').to_series().to_numpy())
            print()
            raise e

        return potential_missing_values_df

    @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_not(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]) -> typing.Tuple[pl.DataFrame, typing.Set[str]]:
        # raise PredicateNotImplementedException(f'Omitting `not` for now')
        try:
            predicate_df, used_variables = self._inner_filter(predicate["not_args"], mapping)  # type: ignore
        except PredicateNotImplementedException as e:
            raise PredicateNotImplementedException(f"Sub-predicate of the not ({e.args}) was not implemented")

        if DEBUG: start = time.perf_counter()

        relevant_vars = list(used_variables)
        potential_missing_values_df = self._build_potential_missing_values_df(mapping, relevant_vars)

        # Now, for each possible combination of args on each trace / domain, 'intervals' will contain the truth intervals if
        # they exist and null otherwise, and 'intervals_right' will contain the empty interval'
        join_columns = ["trace_id"]
        if "trace_length" in predicate_df.columns:
            join_columns.append("trace_length")

        predicate_df = predicate_df.join(potential_missing_values_df, how="outer", on=join_columns + list(used_variables & set(predicate_df.columns)))


        # With the simplified approach, we only want to select the rows where the intervals are null
        # Or rows where the sum of the interval is not the trace length (== one intervals, which is the entire trace)
        # To basically eliminate predicates that are live for the entire trace
        predicate_df = predicate_df.filter(pl.col("intervals").is_null() | \
                                           pl.col("intervals").list.lengths() == 0 | \
                                           ~(pl.col("intervals").list.first().list.sum() == pl.col("trace_length")))

        # Fulling null values instead of union-ing with the null intervals because it's much faster (.apply is really expensive)
        # Invert intervals will then flip them to be the entire length of the trace
        # predicate_df.replace("intervals", predicate_df.with_columns(pl.col("intervals").fill_null(value=[])).select("intervals", "trace_length").apply(self._invert_intervals_tuple_apply, INTERVALS_LIST_POLARS_TYPE)["column_0"])
        predicate_df = predicate_df.drop([c for c in predicate_df.columns if c.endswith("_right")] + ['trace_length'])

        if DEBUG:
            end = time.perf_counter()
            print(f"Time to NOT: {'%.5f' % (end - start)}s")  # type: ignore

        return predicate_df, used_variables

    def _inner_filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]) -> typing.Tuple[pl.DataFrame, typing.Set[str]]:
        '''
        Filters the data by the given predicate and mapping, returning a list of intervals in which the predicate is true
        for each processed trace

        Returns a dictionary mapping from the trace ID to a dictionary that maps from the set of arguments to a list of
        intervals in which the predicate is true for that set of arguments
        '''

        predicate_rule = predicate.parseinfo.rule  # type: ignore

        # Temporarily disable caching to profile without it
        # if predicate_rule in self.cache_rules:
        #     ast_str = ast_section_to_string(predicate, PREFERENCES)
        #     ast_str = re.sub(r"\s+", " ", ast_str)
        #     for key, val in mapping.items():
        #         ast_str = ast_str.replace(key, str(val))

        #     if ast_str in self.predicate_interval_cache:
        #         return self.predicate_interval_cache[ast_str]

        if predicate_rule == "predicate":
            return self._handle_predicate(predicate, mapping)

        elif predicate_rule == "super_predicate":
            return self._inner_filter(predicate["pred"], mapping)  # type: ignore

        elif predicate_rule == "super_predicate_and":
            return self._handle_and(predicate, mapping)

        elif predicate_rule == "super_predicate_or":
            return self._handle_or(predicate, mapping)

        elif predicate_rule == "super_predicate_not":
            return self._handle_not(predicate, mapping)

        elif predicate_rule in ["super_predicate_exists", "super_predicate_forall", "function_comparison"]:
            raise PredicateNotImplementedException(predicate_rule)

        # elif predicate_rule == "super_predicate_exists":
        #     variable_type_mapping = extract_variable_type_mapping(predicate["exists_vars"]["variables"])  # type: ignore

        #     variables = extract_variables(predicate)
        #     unused_variables = [var for var in mapping.keys() if var not in variables]
        #     unused_variable_types = [mapping[var] for var in unused_variables]

        #     interval_mapping = defaultdict(lambda: defaultdict(list))
        #     sub_intervals = self._inner_filter(predicate["exists_args"], {**mapping, **variable_type_mapping}, used_variables)

        #     # Groups the intervals by the part of the mapping that *isn't* within the (exists)
        #     def keyfunc(element):
        #         key = tuple(sorted(elem for elem in element if elem.split('->')[0] not in variable_type_mapping.keys()))
        #         return key

        #     for id in sub_intervals:
        #         sorted_arg_ids = sorted(sub_intervals[id].keys(), key=keyfunc)
        #         for key, group in groupby(sorted_arg_ids, keyfunc):

        #             used_variables = tuple(elem.split('->')[0] for elem in key)
        #             used_objects = tuple(elem.split('->')[1] for elem in key)

        #             # As with [or], above, we need to compute the union of the indices in which the sub-predicate is true
        #             truth_idxs = [self._intervals_to_indices(sub_intervals[id][arg_ids]) for arg_ids in group]
        #             union = set.union(*truth_idxs)

        #             if len(union) > 0:

        #                 domain = self._domain_key(self.data[self.data["id"] == id]["domain"].unique()[0])
        #                 other_object_assignments = get_object_assignments(domain, unused_variable_types, used_objects=used_objects)
        #                 if len(other_object_assignments) == 0:
        #                     other_object_assignments = [()]

        #                 for assignment in other_object_assignments:
        #                     full_assignment = tuple(sorted([f"{var}->{id}" for var, id in zip(used_variables, used_objects)] +
        #                                                    [f"{var}->{id}" for var, id in zip(unused_variables, assignment)]))


        #                     interval_mapping[id][full_assignment] = self._indices_to_intervals(union)

        #     return interval_mapping

        # elif predicate_rule == "super_predicate_forall":
        #     variable_type_mapping = extract_variable_type_mapping(predicate["forall_vars"]["variables"])  # type: ignore

        #     variables = extract_variables(predicate)
        #     unused_variables = [var for var in mapping.keys() if var not in variables]
        #     unused_variable_types = [mapping[var] for var in unused_variables]

        #     interval_mapping = defaultdict(lambda: defaultdict(list))
        #     sub_intervals = self._inner_filter(predicate["forall_args"], {**mapping, **variable_type_mapping}, used_variables)

        #     # Groups the intervals by the part of the mapping that *isn't* within the (forall)
        #     def keyfunc(element):
        #         key = tuple(sorted(elem for elem in element if elem.split('->')[0] not in variable_type_mapping.keys()))
        #         return key

        #     for id in sub_intervals:
        #         sorted_arg_ids = sorted(sub_intervals[id].keys(), key=keyfunc)
        #         for key, group in groupby(sorted_arg_ids, keyfunc):

        #             used_variables = tuple(elem.split('->')[0] for elem in key)
        #             used_objects = tuple(elem.split('->')[1] for elem in key)

        #             # TODO

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")

        # Temporarily disable caching
        # if predicate_rule in self.cache_rules:
        #     self.predicate_interval_cache[ast_str] = interval_mapping




CURRENT_TEST_TRACE_NAMES = [
    '1HOTuIZpRqk2u1nZI1v1-gameplay-attempt-1-rerecorded',
    'IvoZWi01FO2uiNpNHyci-createGame-rerecorded',
    '4WUtnD8W6PGVy0WBtVm4-gameplay-attempt-1-rerecorded',
    'LTZh4k4THamxI5QJfVrk-gameplay-attempt-1-rerecorded',
    'WtZpe3LQFZiztmh7pBBC-gameplay-attempt-1-rerecorded',
    'FyGQn1qJCLTLU1hfQfZ2-preCreateGame-rerecorded',
    '6ZjBeRCvHxG05ORmhInj-gameplay-attempt-1-rerecorded',
    'Tcfpwc8v8HuKRyZr5Dyc-gameplay-attempt-2-rerecorded',
    '4WUtnD8W6PGVy0WBtVm4-createGame-rerecorded',
    '39PytL3fAMFkYXNoB5l6-gameplay-attempt-1-rerecorded',
    '5lTRHBueXsaOu9yhvOQo-gameplay-attempt-1-rerecorded',
    'SQErBa5s5TPVxmm8R6ks-freePlay-rerecorded',
    '9C0wMm4lzrJ5JeP0irIu-preCreateGame-rerecorded',
    'f2WUeVzu41E9Lmqmr2FJ-preCreateGame-rerecorded',
    '6XD5S6MnfzAPQlsP7k30-gameplay-attempt-2-rerecorded',
    'xMUrxzK3fXjgitdzPKsm-freePlay-rerecorded',
    'IhOkh1l3TBY9JJVubzHx-gameplay-attempt-1-rerecorded',
    'WtZpe3LQFZiztmh7pBBC-createGame-rerecorded',
    'vfh1MTEQorWXKy8jOP1x-gameplay-attempt-2-rerecorded',
    'LTZh4k4THamxI5QJfVrk-preCreateGame-rerecorded',
    '79X7tsrbEIu5ffDGnY8q-gameplay-attempt-1-rerecorded',
    'jCc0kkmGUg3xUmUSXg5w-gameplay-attempt-1-rerecorded',
    'ktwB7wT09sh4ivNme3Dw-createGame-rerecorded',
    'ktwB7wT09sh4ivNme3Dw-preCreateGame-rerecorded',
    'ktwB7wT09sh4ivNme3Dw-gameplay-attempt-1-rerecorded',
    'vfh1MTEQorWXKy8jOP1x-createGame-rerecorded',
    'QclKeEZEVr7j0klPuanE-gameplay-attempt-1-rerecorded',
    'jCc0kkmGUg3xUmUSXg5w-preCreateGame-rerecorded',
    'FyGQn1qJCLTLU1hfQfZ2-freePlay-rerecorded',
    'SQErBa5s5TPVxmm8R6ks-editGame-rerecorded',
    'IvoZWi01FO2uiNpNHyci-freePlay-rerecorded',
    'IvoZWi01FO2uiNpNHyci-preCreateGame-rerecorded',
    'SQErBa5s5TPVxmm8R6ks-preCreateGame-rerecorded',
    '9dQSLmtxxIBy0Rsnc8uu-freePlay-rerecorded',
    'FyGQn1qJCLTLU1hfQfZ2-createGame-rerecorded',
    'IhOkh1l3TBY9JJVubzHx-freePlay-rerecorded',
    '7r4cgxJHzLJooFaMG1Rd-gameplay-attempt-1-rerecorded',
    '79X7tsrbEIu5ffDGnY8q-preCreateGame-rerecorded',
    '6XD5S6MnfzAPQlsP7k30-freePlay-rerecorded',
    '9C0wMm4lzrJ5JeP0irIu-gameplay-attempt-1-rerecorded',
    'vfh1MTEQorWXKy8jOP1x-preCreateGame-rerecorded',
    'QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-2-rerecorded',
    'SQErBa5s5TPVxmm8R6ks-gameplay-attempt-1-rerecorded',
    'NJUY0YT1Pq6dZXsmw0wE-gameplay-attempt-2-rerecorded',
    '9dQSLmtxxIBy0Rsnc8uu-createGame-rerecorded',
    'IvoZWi01FO2uiNpNHyci-gameplay-attempt-2-rerecorded',
    'f2WUeVzu41E9Lmqmr2FJ-gameplay-attempt-1-rerecorded',
    'QclKeEZEVr7j0klPuanE-gameplay-attempt-3-rerecorded',
    '4WUtnD8W6PGVy0WBtVm4-freePlay-rerecorded',
    '9dQSLmtxxIBy0Rsnc8uu-gameplay-attempt-1-rerecorded',
    '6XD5S6MnfzAPQlsP7k30-preCreateGame-rerecorded',
    'Tcfpwc8v8HuKRyZr5Dyc-createGame-rerecorded',
    'Tcfpwc8v8HuKRyZr5Dyc-gameplay-attempt-1-rerecorded',
    '5lTRHBueXsaOu9yhvOQo-preCreateGame-rerecorded',
    'IhOkh1l3TBY9JJVubzHx-createGame-rerecorded',
    'QclKeEZEVr7j0klPuanE-gameplay-attempt-2-rerecorded',
    'Tcfpwc8v8HuKRyZr5Dyc-preCreateGame-rerecorded',
    'R9nZAvDq7um7Sg49yf8T-preCreateGame-rerecorded',
    '7r4cgxJHzLJooFaMG1Rd-createGame-rerecorded',
    'QyX7AlBzBW8hZHsJeDWI-preCreateGame-rerecorded',
    'R9nZAvDq7um7Sg49yf8T-gameplay-attempt-1-rerecorded',
    '1HOTuIZpRqk2u1nZI1v1-preCreateGame-rerecorded',
    'xMUrxzK3fXjgitdzPKsm-gameplay-attempt-1-rerecorded',
    'FyGQn1qJCLTLU1hfQfZ2-gameplay-attempt-1-rerecorded',
    '9C0wMm4lzrJ5JeP0irIu-createGame-rerecorded',
    '4WUtnD8W6PGVy0WBtVm4-editGame-rerecorded',
    'NJUY0YT1Pq6dZXsmw0wE-preCreateGame-rerecorded',
    '4WUtnD8W6PGVy0WBtVm4-preCreateGame-rerecorded',
    'xMUrxzK3fXjgitdzPKsm-preCreateGame-rerecorded',
    'NJUY0YT1Pq6dZXsmw0wE-createGame-rerecorded',
    '6ZjBeRCvHxG05ORmhInj-preCreateGame-rerecorded',
    '39PytL3fAMFkYXNoB5l6-createGame-rerecorded',
    'QyX7AlBzBW8hZHsJeDWI-gameplay-attempt-3-rerecorded',
    'f2WUeVzu41E9Lmqmr2FJ-createGame-rerecorded',
    '79X7tsrbEIu5ffDGnY8q-createGame-rerecorded',
    'jCc0kkmGUg3xUmUSXg5w-gameplay-attempt-2-rerecorded',
    '7r4cgxJHzLJooFaMG1Rd-preCreateGame-rerecorded'
]


def _print_results_as_expected_intervals(filter_results):
    print(' ' * 8 + 'expected_intervals={')
    for key, intervals in filter_results.items():
        print(f'{" " * 12}{key}: {intervals},')
    print(' ' * 8 + '}')


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

    stats = CommonSensePredicateStatisticsSplitArgs(cache_dir=DEFAULT_CACHE_DIR,
                                                    # trace_names=CURRENT_TEST_TRACE_NAMES,
                                                    trace_names=FULL_PARTICIPANT_TRACE_SET,
                                                    cache_rules=[],
                                                    base_trace_path=DEFAULT_BASE_TRACE_PATH,
                                                    overwrite=False
                                                    )

    all_object_types = set()
    for room_type in OBJECTS_BY_ROOM_AND_TYPE.keys():
        all_object_types.update(OBJECTS_BY_ROOM_AND_TYPE[room_type].keys())
        all_object_types.update(SPECIFIC_NAMED_OBJECTS_BY_ROOM[room_type].keys())

    all_object_types.update(COLORS)
    all_object_types.update(ORIENTATIONS)
    all_object_types.update(SIDES)

    all_object_types.difference_update(META_TYPES.keys())

    # variable_type_usage = json.loads(open(f"{get_project_dir()}/reward-machine/caches/variable_type_usage.json", "r").read())
    for var_type in sorted(all_object_types):
        if var_type in META_TYPES:
            continue
        var_intervals = stats.data.filter((pl.col("arg_1_type") == var_type) | (pl.col("arg_2_type") == var_type))

        prefix = "[+]" if len(var_intervals) > 0 else "[-]"
        print(f"{prefix} {var_type} has {len(var_intervals)} appearances")

    data_df = pd.read_pickle(stats.stats_filename)
    create_bitstings_df(
        data_df, stats.trace_lengths_and_domains,
        os.path.join(DEFAULT_CACHE_DIR, BITSTRING_INTERVALS_FILE_NAME_FORMAT.format(traces_hash=stats.trace_names_hash)))

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
