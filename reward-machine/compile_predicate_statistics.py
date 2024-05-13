from collections import defaultdict
from functools import reduce
import glob
from itertools import groupby, permutations, product, repeat, starmap
import json
import os
import pandas as pd
import pathlib
import pickle
import polars as pl
import re
import tatsu, tatsu.ast, tatsu.grammars
import time
from tqdm import tqdm
import typing
from viztracer import VizTracer


from config import COLORS, META_TYPES, OBJECTS_BY_ROOM_AND_TYPE, ORIENTATIONS, SIDES, UNITY_PSEUDO_OBJECTS
from utils import (extract_predicate_function_name,
                   extract_variables,
                   extract_variable_type_mapping,
                   get_project_dir,
                   get_object_assignments,
                   FullState)
from ast_parser import PREFERENCES
from ast_printer import ast_section_to_string
from predicate_handler import PREDICATE_LIBRARY_RAW

COMMON_SENSE_PREDICATES_AND_FUNCTIONS = (
    ("adjacent", 2),
    ("agent_holds", 1),
    ("in", 2),
    ("in_motion", 1),
    ("on", 2),
    ("touch", 2),
    # ("between", 3),
)

TYPE_REMAP = {
    "hexagonal_bin": "garbagecan", "bridge_block": "bridgeblock", "cube_block": "cubeblock",
    "cylindrical_block": "cylinderblock", "flat_block": "flatrectblock",
    "pyramid_block": "pyramidblock", "tall_cylindrical_block": "longcylinderblock",
    "tall_rectangular_block": "tallrectblock", "triangle_block": "triangleblock"
}
DEBUG = False
PROFILE = False


class CommonSensePredicateStatistics():
    def __init__(self,
                 cache_dir: typing.Union[str, pathlib.Path],
                 trace_paths: typing.Optional[typing.Sequence[str]] = None,
                 cache_rules: typing.Optional[typing.Sequence[str]] = None,
                 overwrite: bool=False):

        self.cache_dir = cache_dir

        # TODO: swap to .pkl.gz
        stats_filename = os.path.join(cache_dir, 'predicate_statistics.pkl')
        trace_lengths_filename = os.path.join(cache_dir, 'trace_lengths.pkl')

        if os.path.exists(stats_filename) and not overwrite:
            self.data = pd.read_pickle(stats_filename)
            print(f'Loaded data with shape {self.data.shape} from {stats_filename}')
            self.trace_lengths = pickle.load(open(trace_lengths_filename, 'rb'))

        else:
            if trace_paths is None:
                raise ValueError("Must provide trace paths if cache does not exist")

            self.data = pd.DataFrame(columns=['predicate', 'arg_ids', 'arg_types', 'trace_id', 'domain', 'intervals'])
            self.trace_lengths = {}
            for trace_path in tqdm(trace_paths, desc="Processing traces"):
                trace = json.load(open(trace_path, 'r'))
                self.process_trace(trace)

            self.data.to_pickle(stats_filename)
            pickle.dump(self.trace_lengths, open(trace_lengths_filename, 'wb'))

        # Convert to polars and combine the arg types field for easier selection
        def join_args(args):
            return "-".join(args)
        self.data["arg_types"] = self.data["arg_types"].apply(join_args)
        self.data = pl.from_pandas(self.data)

        # Cache calls to get_object_assignments
        self.cache_rules = cache_rules
        self.object_assignment_cache = {}
        self.predicate_interval_cache = {}

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

    def _get_room_objects(self, trace) -> set:
        '''
        Returns the set of objects in the room type of the given trace, excluding pseudo-objects,
        colors, and the agent
        '''

        room_type = self._domain_key(trace['scene'])
        room_objects = set(sum([list(OBJECTS_BY_ROOM_AND_TYPE[room_type][obj_type]) for obj_type in OBJECTS_BY_ROOM_AND_TYPE[room_type]], []))
        room_objects -= set(UNITY_PSEUDO_OBJECTS.keys())
        room_objects -= set(COLORS)
        room_objects -= set(SIDES)
        room_objects -= set(ORIENTATIONS)
        room_objects -= set(['agent'])

        return room_objects

    def _object_assignments(self, domain, variable_types, used_objects=[]):
        '''
        Wrapper around get_object_assignments in order to cache outputs
        '''

        key = (domain, tuple(variable_types), tuple(used_objects))
        if key not in self.object_assignment_cache:
            object_assignments = get_object_assignments(domain, variable_types, used_objects=used_objects)
            self.object_assignment_cache[key] = object_assignments

        return self.object_assignment_cache[key]

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

    def _invert_intervals(self, intervals: typing.List[typing.List[int]], length: int):
        if intervals == []:
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

        received_full_update = False

        full_trace_id = f"{trace['id']}-{trace['replayKey']}"

        for idx, state in tqdm(enumerate(replay), total=replay_len, desc=f"Processing replay {full_trace_id}", leave=False):
            is_final = idx == replay_len - 1
            state = FullState.from_state_dict(state)

            # Track changes to the agent
            if state.agent_state_changed:
                most_recent_agent_state = state.agent_state

            # And to objects
            for obj in state.objects:
                most_recent_object_states[obj.object_id] = obj

            # Check if we've received a full state update, which we detect by seeing if the most_recent_object_states
            # includes every object in the room
            if not received_full_update:
                difference = room_objects.difference(set(most_recent_object_states.keys()))
                received_full_update = (len(difference) == 0)

            # Only perform predicate checks if we've received at least one full state update
            if received_full_update and state.n_objects_changed > 0:
                for predicate, n_args in COMMON_SENSE_PREDICATES_AND_FUNCTIONS:

                    # Some predicates take only an empty list for arguments
                    if n_args == 0:
                        possible_args = [[]]

                    # Collect all possible sets of arguments in which at least one has been updated this step
                    else:
                        changed_this_step = [obj.object_id for obj in state.objects]
                        possible_args = list(product(*([changed_this_step] + list(repeat(room_objects, n_args - 1)))))

                        # Filter out any sets of arguments with duplicates
                        possible_args = [arg_set for arg_set in possible_args if len(set(arg_set)) == len(arg_set)]

                    for arg_set in possible_args:
                        for arg_ids in permutations(arg_set):
                            args = [most_recent_object_states[obj_id] for obj_id in arg_ids]
                            arg_types = tuple([obj.object_type.lower() for obj in args])

                            key = self._predicate_key(predicate, arg_ids)
                            predicate_fn = PREDICATE_LIBRARY_RAW[predicate]

                            evaluation = predicate_fn(most_recent_agent_state, args)

                            # If the predicate is true, then check to see if the last interval is closed. If it is, then
                            # create a new interval
                            if evaluation:
                                if key not in predicate_satisfaction_mapping:
                                    info = {"predicate": predicate, "arg_ids": arg_ids, "arg_types": arg_types,
                                            "trace_id": full_trace_id, "domain": self._domain_key(trace['scene']),
                                            "intervals": [[idx, None]]}
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
        self.trace_lengths[full_trace_id] = replay_len

        # Collapse the intervals into a single dataframe and add it to the overall dataframe
        game_df = pd.DataFrame(predicate_satisfaction_mapping.values())
        self.data = pd.concat([self.data, game_df], ignore_index=True)

    def filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]):
        used_variables = set()
        filter_results = self._inner_filter(predicate, mapping, used_variables)
        return {(key[0], tuple(sorted([key_mapping for key_mapping in key[1] if any(key_mapping.startswith(v) for v in used_variables)]))): intervals
            for key, intervals in filter_results.items()
        }

    def _inner_filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], used_variables: typing.Set[str]):
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
            predicate_name = extract_predicate_function_name(predicate)  # type: ignore
            variables = extract_variables(predicate)  # type: ignore
            used_variables.update(variables)

            # Restrict the mapping to just the referenced variables and expand meta-types + support for directly quantified objects
            relevant_arg_mapping = {}
            for var in variables:
                if var in mapping:
                    relevant_arg_mapping[var] = sum([META_TYPES.get(arg_type, [arg_type]) for arg_type in mapping[var]], [])
                elif not var.startswith("?"):
                    relevant_arg_mapping[var] = [var]

            # Apply the type remapping, when needed. TODO: this is probably pretty slow -- best would be to just be consistent about
            # type naming
            # TODO (gd1279): we should move this to the preprocessing step -- we don't care if it's slow then?
            for key, val in relevant_arg_mapping.items():
                relevant_arg_mapping[key] = [TYPE_REMAP.get(arg_type, arg_type) for arg_type in val]

            # TODO (gd1279): check what happens with unquantifiec objects (e.g. desk)

            # For each predicate, we imagine that it is satisfied for every possible assignment of its unused arguments
            # (e.g., in the context of (?b - ball, ?h - bin), if the predicate (in_motion ?b) is satisfied for ?b = Dodgeball1,
            #  then we store its satisfaction for all possible assignments of ?h). The downside of this is that we wind up
            # storing an interval for every entry in the cartesian product of the unused variables' types -- possible TODO?
            unused_variables = [var for var in mapping.keys() if var not in variables]
            unused_variable_types = tuple([tuple(mapping[var]) for var in unused_variables])

            # In cases where a variable can have multiple types, we consider all possible combinations of types
            possible_arg_types = ["-".join(entry) for entry in product(*[types for types in relevant_arg_mapping.values()])]

            predicate_df = self.data.filter((pl.col("predicate") == predicate_name) & (pl.col("arg_types").is_in(possible_arg_types)))

            all_trace_ids, domains, all_arg_ids, all_intervals = predicate_df["trace_id"], predicate_df["domain"], predicate_df["arg_ids"], predicate_df["intervals"]
            unused_object_assignments = starmap(self._object_assignments, zip(domains, repeat(unused_variable_types), all_arg_ids))

            start = time.perf_counter()
            interval_mapping = defaultdict(list)
            for trace_id, arg_ids, assignments, intervals in zip(all_trace_ids, all_arg_ids, unused_object_assignments, all_intervals):
                for assignment in assignments:
                    full_variables = variables + unused_variables
                    full_objects = tuple(arg_ids.to_list()) + assignment

                    full_assignment = tuple(sorted([f"{var}->{id}" for var, id in zip(full_variables, full_objects)]))

                    interval_mapping[(trace_id, full_assignment)] = intervals.to_list()

            end = time.perf_counter()
            if DEBUG: print(f"Time per collect '{predicate_name}': {'%.5f' % (end - start)}s")

        elif predicate_rule == "super_predicate":
            interval_mapping = self._inner_filter(predicate["pred"], mapping, used_variables)  # type: ignore

        elif predicate_rule == "super_predicate_and":
            and_args = predicate["and_args"]
            if isinstance(and_args, tatsu.ast.AST):
                and_args = [and_args]

            interval_mapping = defaultdict(lambda: defaultdict(list))
            sub_interval_mappings = [self._inner_filter(and_arg, mapping, used_variables) for and_arg in and_args]  # type: ignore

            start = time.perf_counter()
            keys = set(sum([list(sub_interval_mapping.keys()) for sub_interval_mapping in sub_interval_mappings], []))

            # TODO (gd1279): would this fail if we disjuncted more than two predicates?
            interval_mapping = {key: reduce(self._intersect_intervals, [sub[key] for sub in sub_interval_mappings]) for key in keys}

            # Filter out empty intervals
            interval_mapping = defaultdict(list, {key: val for key, val in interval_mapping.items() if val != []})

            end = time.perf_counter()
            if DEBUG: print(f"Time to intersect: {'%.5f' % (end - start)}s")

        elif predicate_rule == "super_predicate_or":
            or_args = predicate["or_args"]
            if isinstance(or_args, tatsu.ast.AST):
                or_args = [or_args]

            interval_mapping = defaultdict(lambda: defaultdict(list))
            sub_interval_mappings = [self._inner_filter(or_arg, mapping, used_variables) for or_arg in or_args]  # type: ignore

            keys = set(sum([list(sub_interval_mapping.keys()) for sub_interval_mapping in sub_interval_mappings], []))

            # TODO (gd1279): would this fail if we conjuncted more than two predicates?
            interval_mapping = {key: reduce(self._union_intervals, [sub[key] for sub in sub_interval_mappings]) for key in keys}

            # Filter out empty intervals
            # TODO (gd1279): is there ever a reason that a union of two intervals would come up as empty?
            interval_mapping = defaultdict(list, {key: val for key, val in interval_mapping.items() if val != []})

        elif predicate_rule == "super_predicate_not":
            sub_intervals = self._inner_filter(predicate["not_args"], mapping, used_variables)  # type: ignore


            start = time.perf_counter()
            interval_mapping = defaultdict(list)

            # We need to check every trace ID in the dataset in case there are some traces in which the sub-predicate is never true
            for trace_id, length in self.trace_lengths.items():

                domain = self.data.filter(pl.col("trace_id") == trace_id)["domain"][0]

                variable_types = tuple(tuple(mapping[var]) for var in mapping.keys())
                possible_arg_assignments = self._object_assignments(domain, variable_types)

                # Similarly, we need to check every possible set of arguments in case there are some sets in which the sub-predicate is never true
                for arg_ids in possible_arg_assignments:

                    argument_mapping = tuple(sorted(f"{var}->{id}" for var, id in zip(mapping.keys(), arg_ids)))
                    key = (trace_id, argument_mapping)
                    interval_mapping[key] = self._invert_intervals(sub_intervals[key], length)

            end = time.perf_counter()
            if DEBUG: print(f"Time to invert: {'%.5f' % (end - start)}s")

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
        #     sub_intervals = self._inner_filter(predicate["forall_args"], {**mapping, **variable_type_mapping})

        #     # Groups the intervals by the part of the mapping that *isn't* within the (forall)
        #     def keyfunc(element):
        #         key = tuple(sorted(elem for elem in element if elem.split('->')[0] not in variable_type_mapping.keys()))
        #         return key

        #     for id in sub_intervals:
        #         sorted_arg_ids = sorted(sub_intervals[id].keys(), key=keyfunc)
        #         for key, group in groupby(sorted_arg_ids, keyfunc):

        #             used_variables = tuple(elem.split('->')[0] for elem in key)
        #             used_objects = tuple(elem.split('->')[1] for elem in key)

                    # TODO

        else:
            raise ValueError(f"Error: Unknown rule '{predicate_rule}'")

        # Temporarily disable caching
        # if predicate_rule in self.cache_rules:
        #     self.predicate_interval_cache[ast_str] = interval_mapping

        return interval_mapping


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

    # should be: (and (in_motion ?b) (not (agent_holds ?b)))
    test_pred_1 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][1]['seq_func']['hold_pred']

    # should be: (and (not (in_motion ?b)) (in ?h ?b)))
    test_pred_2 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][2]['seq_func']['once_pred']

    # should be: (once (and (not (in_motion ?b) (exists (?c - hexagonal_bin) (in ?c ?b)))))
    test_pred_3 = game_ast[4][1]['preferences'][0]['definition']['forall_pref']['preferences']['pref_body']['body']['exists_args']['then_funcs'][3]['seq_func']['once_pred']

    block_stacking_game = open(get_project_dir() + '/reward-machine/games/block_stacking.txt').read()
    block_stacking_game_ast = grammar_parser.parse(block_stacking_game)  # type: ignore

    test_pred_or = block_stacking_game_ast[3][1]['preferences'][0]['definition']['pref_body']['body']['exists_args']['then_funcs'][2]['seq_func']['hold_pred']

    test_pred_desk_or = block_stacking_game_ast[3][1]['preferences'][1]['definition']['pref_body']['body']['exists_args']['at_end_pred']


    test_mapping = {"?b": ["ball"], "?h": ["hexagonal_bin"]}
    block_test_mapping = {"?b1": ['cube_block'], "?b2": ["cube_block"]}
    block_desk_test_mapping = {"?b": ["block"]}
    all_block_test_mapping = {"?b1": ["block"], "?b2": ["block"]}


    trace_path = pathlib.Path(get_project_dir() + '/reward-machine/traces/three_wall_to_bin_bounces-RErerecorded.json')
    cache_dir = pathlib.Path(get_project_dir() + '/reward-machine/caches')
    # stats = CommonSensePredicateStatistics(cache_dir, [trace_path], overwrite=True)

    # TEST_TRACE_NAMES = ["throw_ball_to_bin_unique_positions", "setup_test_trace", "building_castle",
    #                     "throw_all_dodgeballs", "stack_3_cube_blocks", "three_wall_to_bin_bounces",
    #                     "complex_stacking_trace"]

    # TEST_TRACE_NAMES = ["ZBcXIZbvTS3U4IBGk1zk-createGame-rerecorded",
    #                     "ZBcXIZbvTS3U4IBGk1zk-preCreateGame-rerecorded",
    #                     "KO8pbUWEpZldxy7AzyM5-gameplay-attempt-1-rerecorded",
    #                     "KO8pbUWEpZldxy7AzyM5-createGame-rerecorded",
    #                     "KO8pbUWEpZldxy7AzyM5-preCreateGame-rerecorded",
    #                     "c4bea3VqKksZ7Rd5RdTO-gameplay-attempt-1-rerecorded",
    #                     "c4bea3VqKksZ7Rd5RdTO-preCreateGame-rerecorded",
    #                     "ZMqZkrMMB0PcsCeLhQqE-gameplay-attempt-1-rerecorded",
    #                     "ZMqZkrMMB0PcsCeLhQqE-preCreateGame-rerecorded",
    #                     "three_wall_to_bin_bounces-RErerecorded"]

    # trace_paths = [f"{get_project_dir()}/reward-machine/traces/{trace}.json" for trace in TEST_TRACE_NAMES]

    trace_paths = glob.glob(f"{get_project_dir()}/reward-machine/traces/participant-traces/*.json")

    # trace_paths = [f"{get_project_dir()}/reward-machine/traces/participant-traces/{trace}" for trace in
    #                 os.listdir(f"{get_project_dir()}/reward-machine/traces/participant-traces")]

    stats = CommonSensePredicateStatistics(cache_dir, trace_paths, cache_rules=["predicate", "super_predicate_and",
                                           "super_predicate_or", "super_predicate_not"], overwrite=False)
    tracer = None
    if PROFILE:
        tracer = VizTracer(10000000)
        tracer.start()

    test_out = stats.filter(test_pred_desk_or, block_desk_test_mapping)
    # _print_results_as_expected_intervals(test_out)
    start = time.perf_counter()
    N_ITER = 10
    for i in range(N_ITER):
        # print(f"\n====================")
        inner_start = time.perf_counter()
        stats.filter(test_pred_or, all_block_test_mapping)
        # inner_end = time.perf_counter()
        # print(f"Time per iteration: {'%.5f' % (inner_end - inner_start)}s")
    end = time.perf_counter()
    print(f"Time per iteration: {'%.5f' % ((end - start) / N_ITER)}s")

    if tracer is not None:
        tracer.stop()
        profile_output_path = os.path.join(get_project_dir(), 'reward-machine/temp/viztracer.json')
        print(f'Saving profile to {profile_output_path}')
        tracer.save(profile_output_path)

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
