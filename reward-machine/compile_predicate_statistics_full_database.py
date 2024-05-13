from collections import defaultdict
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
import platform
import polars as pl
pl.enable_string_cache(True)
import pstats
import shutil
import signal
import tatsu, tatsu.ast, tatsu.grammars
import time
from tqdm import tqdm
import typing
from viztracer import VizTracer


from config import ROOMS, META_TYPES, TYPES_TO_META_TYPE, OBJECTS_BY_ROOM_AND_TYPE, ORIENTATIONS, SIDES, UNITY_PSEUDO_OBJECTS, NAMED_WALLS, SPECIFIC_NAMED_OBJECTS_BY_ROOM, OBJECT_ID_TO_SPECIFIC_NAME_BY_ROOM, GAME_OBJECT, GAME_OBJECT_EXCLUDED_TYPES, BUILDING, BLOCK, ALL_NON_OBJECT_TYPES
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
    ("game_start", 0),
    ("game_over", 0),
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


DEFAULT_CACHE_DIR = pathlib.Path(get_project_dir() + '/reward-machine/caches')
DEFAULT_CACHE_FILE_NAME_FORMAT = 'predicate_statistics_bitstring_intervals_{traces_hash}.pkl.gz'
NO_INTERVALS_CACHE_FILE_NAME_FORMAT = 'predicate_statistics_no_intervals_{traces_hash}.pkl.gz'
DEFAULT_TRACE_LENGTHS_FILE_NAME_FORMAT = 'trace_lengths_{traces_hash}.pkl'
DEFAULT_IN_PROCESS_TRACES_FILE_NAME_FORMAT = 'in_progress_traces_{traces_hash}.pkl'
DEFAULT_BASE_TRACE_PATH = os.path.join(os.path.dirname(__file__), "traces/participant-traces/")
CLUSTER_BASE_TRACE_PATH = '/misc/vlgscratch4/LakeGroup/guy/participant-traces'

if os.path.exists('/scratch/gd1279'):
    DUCKDB_TMP_FOLDER = '/scratch/gd1279/duckdb'
else:
    DUCKDB_TMP_FOLDER = '/tmp/duckdb'
DUCKDB_QUERY_LOG_FOLDER = '/tmp/duckdb_query_logs'



FULL_PARTICIPANT_TRACE_SET = [os.path.splitext(os.path.basename(t))[0] for t in glob.glob(os.path.join(DEFAULT_BASE_TRACE_PATH, '*.json'))]


class PredicateNotImplementedException(Exception):
    pass


class MissingVariableException(Exception):
    pass


class QueryTimeoutException(Exception):
    pass


class Timeout:
    def __init__(self, seconds=1, message="Timed out"):
        self._seconds = seconds
        self._message = message

    @property
    def seconds(self):
        return self._seconds

    @property
    def message(self):
        return self._message

    @property
    def handler(self):
        return self._handler

    @handler.setter
    def handler(self, handler):
        self._handler = handler

    def handle_timeout(self, *_):
        raise QueryTimeoutException(self.message)

    def __enter__(self):
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *_):
        signal.alarm(0)
        pass


def raise_query_timeout(*args, **kwargs):
    raise QueryTimeoutException("Query timed out")


# alarm_handler = signal.getsignal(signal.SIGALRM)
# if alarm_handler is not None and alarm_handler is not signal.SIG_DFL and alarm_handler is not signal.SIG_IGN:
    # signal.signal(signal.SIGALRM, raise_query_timeout)



def stable_hash(str_data: str):
    return hashlib.md5(bytearray(str_data, 'utf-8')).hexdigest()


def stable_hash_list(list_data: typing.Sequence[str]):
    return stable_hash('\n'.join(sorted(list_data)))


DEBUG = False
DEBUG_FINAL_QUERY = False
DEBUG_MISSING_PREDICATE_QUERY = True

MAX_CACHE_SIZE = 2 ** 12
MAX_CHILD_ARGS = 4  # 6
DEFAULT_QUERY_TIMEOUT = 15  # 15  # seconds
IGNORE_PREDICATES = ['equal_x_position', 'equal_z_position']
PREDICATE_TABLE_NAME = 'predicate'
NEGATED_PREDICATE_TABLE_NAME = 'negated_predicate'
MISSING_NEGATED_PREDICATE_TABLE_NAME = 'missing_negated_predicate'

class CommonSensePredicateStatisticsFullDatabase():
    __instance = None

    @staticmethod
    def get_instance(**kwargs):
        if CommonSensePredicateStatisticsFullDatabase.__instance is None:
            CommonSensePredicateStatisticsFullDatabase(**kwargs)

        return CommonSensePredicateStatisticsFullDatabase.__instance


    def __init__(self,
                 cache_dir: typing.Union[str, pathlib.Path] = DEFAULT_CACHE_DIR,
                 cache_filename_format: str = DEFAULT_CACHE_FILE_NAME_FORMAT,
                 trace_lengths_filename_format: str = DEFAULT_TRACE_LENGTHS_FILE_NAME_FORMAT,
                 force_trace_names_hash: typing.Optional[str] = None,
                 trace_folder: typing.Optional[str] = None,
                 max_cache_size: int = MAX_CACHE_SIZE,
                 max_child_args: int = MAX_CHILD_ARGS,
                 query_timeout: int = DEFAULT_QUERY_TIMEOUT,
                 log_queries: bool = False,
                 duckdb_database_str: str = ':memory:',
                 ignore_predicates: typing.Sequence[str] = IGNORE_PREDICATES,
                 ):
        if CommonSensePredicateStatisticsFullDatabase.__instance is not None:
            raise Exception("This class is a singleton!")

        CommonSensePredicateStatisticsFullDatabase.__instance = self

        self.con = duckdb.connect(database=duckdb_database_str)
        self.cache = cachetools.LRUCache(maxsize=max_cache_size)
        self.temp_table_index = -1
        self.temp_table_prefix = 't'
        self.temp_dir = None
        self.max_child_args = max_child_args
        self.query_timeout = query_timeout
        self.hostname = platform.node().split('.', 1)[0]
        self.log_queries = log_queries
        self.ignore_predicates = ignore_predicates

        signal.signal(signal.SIGALRM, raise_query_timeout)

        if trace_folder is None:
            if os.path.exists(CLUSTER_BASE_TRACE_PATH):
                trace_folder = CLUSTER_BASE_TRACE_PATH
            else:
                trace_folder = DEFAULT_BASE_TRACE_PATH

        self.all_trace_ids = [os.path.splitext(os.path.basename(t))[0] for t in glob.glob(os.path.join(trace_folder, '*.json'))]
        self.all_predicates = set([t[0] for t in COMMON_SENSE_PREDICATES_AND_FUNCTIONS if t[0] not in self.ignore_predicates])
        self.all_types = set(reduce(lambda x, y: x + y, [list(x.keys()) for x in chain(OBJECTS_BY_ROOM_AND_TYPE.values(), SPECIFIC_NAMED_OBJECTS_BY_ROOM.values())]))
        self.all_types.difference_update(META_TYPES.keys())
        self.all_types.remove(GAME_OBJECT)
        self.all_types.add(BUILDING)
        self.all_arg_ids = set(reduce(lambda x, y: x + y, [object_types for room_types in OBJECTS_BY_ROOM_AND_TYPE.values() for object_types in room_types.values()]))
        self.all_arg_ids.update(UNITY_PSEUDO_OBJECTS.keys())

        self.game_object_excluded_arg_types = set(GAME_OBJECT_EXCLUDED_TYPES)
        self.game_object_excluded_arg_types.difference_update(META_TYPES.keys())

        self.trace_names_hash = force_trace_names_hash
        self.stats_filename = os.path.join(cache_dir, cache_filename_format.format(traces_hash=self.trace_names_hash))
        self.trace_lengths_and_domains_filename = os.path.join(cache_dir, trace_lengths_filename_format.format(traces_hash=self.trace_names_hash))

        self._create_databases()
        if self.log_queries:
            self._create_query_logger()

    def _create_query_logger(self):
        self.logger = logging.getLogger(f'duckdb_queries_{os.getpid()}')
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        if not os.path.exists(DUCKDB_QUERY_LOG_FOLDER):
            os.makedirs(DUCKDB_QUERY_LOG_FOLDER)

        self.file_handler = logging.FileHandler(f'{DUCKDB_QUERY_LOG_FOLDER}/{os.getpid()}.log')
        self.file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.file_handler)

    # def __getstate__(self) -> typing.Dict[str, typing.Any]:
    #     state = self.__dict__.copy()
    #     return state

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        self.__dict__.update(state)
        self._create_databases()

    def __del__(self):
        # if self.temp_dir is not None and os is not None and os.path.exists(self.temp_dir):
        #     if logger is not None: logger.info(f"Deleting temp directory {self.temp_dir}")
        #     if shutil is not None: shutil.rmtree(self.temp_dir)
        pass

    def _create_databases(self):
        table_query = self.con.execute("SHOW TABLES;")
        if table_query is not None:
            all_tables = set(t.lower() for t in chain.from_iterable(table_query.fetchall()))
            if 'data' in all_tables:
                # logger.info('Skipping creating tables because they already exist')
                return

        logger.info(f"Loading data from files with query timeout {self.query_timeout}")
        open_method = gzip.open if self.stats_filename.endswith('.gz') else open

        if os.path.exists(self.stats_filename):
            data_df = pd.read_pickle(self.stats_filename)

        else:
            raise ValueError(f"Could not find file {self.stats_filename}")

        if os.path.exists(self.trace_lengths_and_domains_filename):
            with open_method(self.trace_lengths_and_domains_filename, 'rb') as f:
                trace_lengths_and_domains = pickle.load(f)
                self.all_trace_ids = list(trace_lengths_and_domains.keys())
                self.trace_id_to_length = {trace_id: length for (trace_id, (length, _)) in trace_lengths_and_domains.items()}

        else:
            raise ValueError(f"Could not find file {self.trace_lengths_and_domains_filename}")

        logger.info("Creating DuckDB table...")
        self.temp_dir = os.path.join(DUCKDB_TMP_FOLDER, self.hostname, str(os.getpid()))
        os.makedirs(self.temp_dir, exist_ok=True)
        self.con.execute(f"SET temp_directory='{self.temp_dir}';")
        self.con.execute("SET enable_progress_bar=false;")
        self.con.execute("SET enable_progress_bar_print=false;")
        self.con.execute("SET memory_limit='32GB';")

        # self.con.execute(f"SET log_query_path='{DUCKDB_TMP_FOLDER}/{os.getpid()}_queries.log';")

        self.con.execute(f"CREATE TYPE domain AS ENUM {tuple(ROOMS)};")
        self.con.execute(f"CREATE TYPE trace_id AS ENUM {tuple(self.all_trace_ids)};")
        self.con.execute(f"CREATE TYPE predicate AS ENUM {tuple(sorted(self.all_predicates))};")
        self.con.execute(f"CREATE TYPE arg_type AS ENUM {tuple(sorted(self.all_types))};")
        self.con.execute(f"CREATE TYPE arg_id AS ENUM {tuple(sorted(self.all_arg_ids))};")

        trace_length_and_domain_rows = [(trace_id, domain, length) for (trace_id, (length, domain)) in trace_lengths_and_domains.items()]
        self.con.execute('CREATE TABLE trace_length_and_domains(trace_id trace_id PRIMARY KEY, domain domain NOT NULL, length INTEGER NOT NULL, intervals BITSTRING);')
        self.con.execute(f'INSERT INTO trace_length_and_domains(trace_id, domain, length) VALUES {str(tuple(trace_length_and_domain_rows))[1:-1]};')
        self.con.execute("UPDATE trace_length_and_domains SET intervals = BITSTRING('0', length);")
        self.con.execute('CREATE INDEX idx_tld_domain ON trace_length_and_domains (domain);')

        # self.con.execute("CREATE TABLE empty_bitstrings(trace_id trace_id PRIMARY KEY, intervals BITSTRING NOT NULL);")
        # self.con.execute("INSERT INTO empty_bitstrings SELECT trace_id, BITSTRING('0', length) as intervals FROM trace_length_and_domains;")

        data_rows = []
        for domain in ROOMS:
            for object_dict in (OBJECTS_BY_ROOM_AND_TYPE[domain], SPECIFIC_NAMED_OBJECTS_BY_ROOM[domain]):
                for object_type, object_ids in object_dict.items():
                    if object_type in self.all_types:
                        for object_id in object_ids:
                            data_rows.append((domain, object_type, object_id, object_type not in self.game_object_excluded_arg_types, object_type in META_TYPES[BLOCK]))

        self.con.execute('CREATE TABLE object_type_to_id(domain domain NOT NULL, type arg_type NOT NULL, object_id arg_id NOT NULL, is_game_object BOOLEAN NOT NULL, is_block BOOLEAN NOT NULL);')
        self.con.execute(f'INSERT INTO object_type_to_id VALUES {str(tuple(data_rows))[1:-1]};')
        # self.con.execute('CREATE INDEX idx_obj_id_domain ON object_type_to_id (domain);')
        # self.con.execute('CREATE INDEX idx_obj_id_type ON object_type_to_id (type);')
        # self.con.execute('CREATE INDEX idx_obj_id_id ON object_type_to_id (object_id);')
        self.con.execute('CREATE INDEX idx_obj_id_joint ON object_type_to_id (type, domain, object_id);')

        data_df = data_df[~data_df.predicate.isin(self.ignore_predicates)]

        self.con.execute("CREATE TABLE data(predicate predicate NOT NULL, arg_1_id arg_id, arg_1_type arg_type, arg_1_is_game_object BOOLEAN, arg_1_is_block BOOLEAN, arg_2_id arg_id, arg_2_type arg_type, arg_2_is_game_object BOOLEAN, arg_2_is_block BOOLEAN, trace_id trace_id NOT NULL, domain domain NOT NULL, intervals BITSTRING NOT NULL);")
        self.con.execute("INSERT INTO data SELECT predicate, arg_1_id, arg_1_type, arg_1_is_game_object, arg_1_is_block, arg_2_id, arg_2_type, arg_2_is_game_object, arg_2_is_block, trace_id, domain, intervals FROM data_df;")

        self.con.execute("INSERT INTO data (predicate, trace_id, domain, intervals) SELECT 'game_start' as predicate, trace_id, domain, set_bit(bitstring('0', length), 0, 1) as intervals FROM trace_length_and_domains;")
        self.con.execute("INSERT INTO data (predicate, trace_id, domain, intervals) SELECT 'game_over' as predicate, trace_id, domain, bitstring('1', length) as intervals FROM trace_length_and_domains;")

        logger.info(f"Creating data table indices...")
        self.con.execute('CREATE INDEX idx_data_trace_id ON data (trace_id);')
        # self.con.execute('CREATE INDEX idx_data_predicate ON data (predicate);')
        self.con.execute('CREATE INDEX idx_data_arg_1_id ON data (arg_1_id);')
        self.con.execute('CREATE INDEX idx_data_arg_2_id ON data (arg_2_id);')
        self.con.execute('CREATE INDEX idx_data_predicate_types ON data (predicate, arg_1_type, arg_2_type);')
        self.con.execute('CREATE INDEX idx_data_predicate_1_type_2_block ON data (predicate, arg_1_type, arg_2_is_block);')
        self.con.execute('CREATE INDEX idx_data_predicate_1_block_2_type ON data (predicate, arg_1_is_block, arg_2_type);')
        self.con.execute('CREATE INDEX idx_data_predicate_1_type_2_go ON data (predicate, arg_1_type, arg_2_is_game_object);')
        self.con.execute('CREATE INDEX idx_data_predicate_1_go_2_type ON data (predicate, arg_1_is_game_object, arg_2_type);')
        self.con.execute('CREATE INDEX idx_data_predicate_1_go_2_go ON data (predicate, arg_1_is_game_object, arg_2_is_game_object);')
        self.con.execute('CREATE INDEX idx_data_predicate_1_block_2_block ON data (predicate, arg_1_is_block, arg_2_is_block);')

        data_rows = self.con.execute("SELECT count(*) FROM data;").fetchone()[0]  # type: ignore
        logger.info(f"Loaded data, found {data_rows} rows")
        del data_df
        del trace_lengths_and_domains

    def _table_name(self, index: int):
        return f"{self.temp_table_prefix}{index}"

    def _next_temp_table_index(self):
        self.temp_table_index += 1
        return self.temp_table_index

    def _next_temp_table_name(self):
        return self._table_name(self._next_temp_table_index())

    def _predicate_and_mapping_cache_key(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], *args, **kwargs) -> str:
        '''
        Returns a string that uniquely identifies the predicate and mapping
        '''
        return f'{ast_section_to_string(predicate, PREFERENCES)}_{str(mapping)}_{str(kwargs)}'

    @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs):
        result_query = None
        try:
            if self.temp_table_index > 2 ** 31:
                self.temp_table_index = -1

            result_query, relevant_variables, is_refactored_implementation = self._inner_filter(predicate, mapping, **kwargs)
            if self.log_queries:
                key = self._predicate_and_mapping_cache_key(predicate, mapping, **kwargs)
                self.logger.info(f'{key}:\n{result_query}\n{"=" * 100}')
                self.file_handler.flush()

            if DEBUG_FINAL_QUERY:
                import sqlparse
                print(sqlparse.format(result_query, reindent=True, keyword_case='upper'))

            with Timeout(seconds=self.query_timeout):
                if 'return_full_result' in kwargs and kwargs['return_full_result']:
                    output_query = f"SELECT DISTINCT(*) FROM ({result_query});"
                    return self.con.execute(output_query).fetchdf()
                if 'return_interval_counts' in kwargs and kwargs['return_interval_counts']:
                    select_variables = ', '.join(f'"{v}"' for v in relevant_variables)
                    output_query = f"SELECT DISTINCT ON(trace_id, domain, {select_variables}) trace_id, domain, bit_count(intervals) AS 'bit_count', {select_variables} FROM ({result_query});"
                    return self.con.execute(output_query).fetchdf()
                if 'last_interval_bit_set' in kwargs and kwargs['last_interval_bit_set']:
                    select_variables = ', '.join(f'"{v}"' for v in relevant_variables)
                    output_query = f"SELECT DISTINCT ON(trace_id, domain, {select_variables}) trace_id, domain, get_bit(intervals, bit_length(intervals)::INTEGER - 1) AS 'last_bit', {select_variables} FROM ({result_query});"
                    return self.con.execute(output_query).fetchdf()
                if 'return_first_state_index' in kwargs and kwargs['return_first_state_index']:
                    select_variables = ', '.join(f'"{v}"' for v in relevant_variables)
                    output_query = f"SELECT DISTINCT ON(trace_id, domain, {select_variables}) trace_id, domain, (bit_position('1'::BIT, intervals) - 1) AS 'first_state_index', {select_variables} FROM ({result_query});"
                    return self.con.execute(output_query).fetchdf()
                elif 'return_trace_ids' in kwargs and kwargs['return_trace_ids']:
                    if is_refactored_implementation:
                        output_query = result_query
                    else:
                        output_query = f"SELECT DISTINCT(trace_id) FROM ({result_query});"
                    return tuple(chain.from_iterable(self.con.execute(output_query).fetchall()))
                else:
                    # if is_refactored_implementation:
                    #     output_query = result_query
                    # else:
                    output_query = f"SELECT COUNT(*) FROM ({result_query} LIMIT 1);"

                    query_result = self.con.execute(output_query)
                    return query_result.fetchall()[0][0]  # type: ignore

        except PredicateNotImplementedException as e:
            # Pass the exception through and let the caller handle it
            raise e

        except duckdb.InvalidInputException as e:
            if result_query is not None:
                logger.error(f"Invalid input exception for query:\n{result_query}")

            raise e

        except QueryTimeoutException:
            logger.warn(f"Query timed out in PID {os.getpid()} for predicate with cache key `{self._predicate_and_mapping_cache_key(predicate, mapping)}`, returning None so a value is cached")
            return None

    def _map_variable_to_type_list(self, var: str, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]) -> typing.List[str]:
        if var in mapping:
            # added an exception fo the generic block type because we handle it with a separate check later
            return sum([META_TYPES.get(arg_type, [arg_type]) if arg_type != BLOCK else [BLOCK] for arg_type in mapping[var]], [])

        # This handles variables which are referenced directly, like the desk and bed
        elif not var.startswith("?"):
            return [var]

        else:
            raise MissingVariableException(f"Variable {var} is not in the mapping")

    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _build_predicate_select_where_items(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]],
                                            table_name: typing.Optional[str] = None, **kwargs) -> typing.Tuple[typing.List[str], typing.Dict[str, str], typing.List[str], typing.Set[str]]:
        predicate_name = extract_predicate_function_name(predicate)  # type: ignore
        # Temporary hack: we don't have `near` implemented yet, but (adjacent ?a ?b) implies (near ?a ?b)
        # So for the time being, we use it as a proxy
        if predicate_name == 'near':
            predicate_name = 'adjacent'

        # Second temporary hack: we don't have `adjacent_side` implemented yet, so we'll take `(adjacent a b)`
        # to be a proxy that `(adjacent_side a s b)` is plausible
        keep_indices = None
        if predicate_name.startswith('adjacent_side'):
            predicate_name = 'adjacent'
            keep_indices = [0, 2]

        if table_name is not None:
            table_name = f"{table_name}."
        else:
            table_name = ""


        if predicate_name not in self.all_predicates:
            raise PredicateNotImplementedException(predicate_name)

        try:
            variables = extract_variables(predicate, error_on_repeat=True)  # type: ignore

        except ValueError:
            logger.warn('Variable repeated in predicate arguments')
            raise MissingVariableException('Variable repeated in predicate arguments')

        if keep_indices is not None:
            if len(variables) <= max(keep_indices):
                pred_str = ''
                try:
                    pred_str = ast_section_to_string(predicate, PREFERENCES)
                except:
                    pass

                logger.warn(f'Argument length mismatch in {predicate_name} with keep indices {keep_indices}: {pred_str}')

                raise MissingVariableException('Not enough variables for the keep indices')

            variables = [variables[i] for i in keep_indices]

        used_variables = set(variables)

        # Restrict the mapping to just the referenced variables and expand meta-types
        relevant_arg_mapping = {var: self._map_variable_to_type_list(var, mapping) for var in variables}

        select_items = [f"{table_name}trace_id", f"{table_name}domain", f"{table_name}intervals"]
        select_dict = {}
        where_items = [f"{table_name}predicate='{predicate_name}'"]

        for i, (arg_var, arg_types) in enumerate(relevant_arg_mapping.items()):
            # if it can be the generic object type or a block, we filter for them specifically
            if GAME_OBJECT in arg_types or BLOCK in arg_types:
                custom_where_clause_components = []
                check_game_object = False
                check_block = False

                if GAME_OBJECT in arg_types:
                    check_game_object = True
                    custom_where_clause_components.append(f"{table_name}arg_{i + 1}_is_game_object IS TRUE")

                if BLOCK in arg_types:
                    check_block = True
                    custom_where_clause_components.append(f"{table_name}arg_{i + 1}_is_block IS TRUE")

                remaining_arg_types = []
                for arg_type in arg_types:
                    include_arg_type = True
                    if check_game_object and (arg_type == GAME_OBJECT or arg_type not in self.game_object_excluded_arg_types):
                        include_arg_type = False

                    if check_block and (arg_type == BLOCK or arg_type in META_TYPES[BLOCK]):
                        include_arg_type = False

                    if include_arg_type:
                        remaining_arg_types.append(arg_type)

                if remaining_arg_types:
                    custom_where_clause_components.append(f"{table_name}arg_{i + 1}_type IN {self._types_to_arg_casts(remaining_arg_types)}")

                if len(custom_where_clause_components) == 1:
                    where_items.append(custom_where_clause_components[0])

                else:
                    custom_where_clause_components = [f"({c})" for c in custom_where_clause_components]
                    where_items.append(f"({' OR '.join(custom_where_clause_components)})")

            else:
                if len(arg_types) == 1:
                    where_items.append(f"({table_name}arg_{i + 1}_type='{arg_types[0]}')")
                else:
                    where_items.append(f"({table_name}arg_{i + 1}_type IN {self._types_to_arg_casts(arg_types)})")

            select_dict[arg_var] = f'{table_name}arg_{i + 1}_id'
            select_items.append(f'{table_name}arg_{i + 1}_id AS "{arg_var}"')

        return select_items, select_dict, where_items, used_variables

    def _build_simple_predicate_query(self, select_items: typing.List[str], where_items: typing.List[str], table_name: typing.Optional[str] = None) -> str:
        if table_name is not None:
            table_name = f"{table_name}."
        else:
            table_name = ""

        return f"SELECT {', '.join(select_items)} FROM data {table_name[:-1]} WHERE {' AND '.join(where_items)}"

    def _handle_predicate(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str], bool]:
        select_items, select_dict, where_items, used_variables = self._build_predicate_select_where_items(predicate, mapping, **kwargs)
        return self._build_simple_predicate_query(select_items, where_items), used_variables, False

    # def _build_query_from_predicate_items(self, predicate_items: typing.List[typing.Tuple[typing.List[str], typing.Dict[str, str], typing.List[str], typing.Set[str], str]],
    #                                       conjunction: bool = True) -> typing.Tuple[str, typing.Set[str]]:

    #     if len(predicate_items) == 1:
    #         select_items, select_dict, where_items, used_variables, table_name = predicate_items[0]
    #         return self._build_simple_predicate_query(select_items, where_items, table_name), used_variables

    #     used_var_to_first_table_name = {}
    #     intervals_logical_op = ' & ' if conjunction else ' | '
    #     join_type = 'INNER JOIN' if conjunction else 'FULL JOIN'
    #     all_select_items, initial_select_dict, all_where_items, all_used_variables, first_table_name = predicate_items[0]

    #     # replace the intervals with the conjunction of all intervals
    #     for col_index, shared_col in enumerate(['trace_id', 'domain']):
    #         shared_col_per_table = [item[0][col_index] for item in predicate_items]
    #         all_select_items[col_index] = f'COALESCE({", ".join(shared_col_per_table)}) AS "{shared_col}"'

    #     intervals_per_table = [item[0][2] for item in predicate_items]
    #     all_select_items[2] = f'({intervals_logical_op.join(intervals_per_table)}) AS intervals'

    #     for v in all_used_variables:
    #         used_var_to_first_table_name[v] = initial_select_dict[v]

    #     join_clauses = []

    #     for select_items, select_dict, where_items, used_variables, table_name in predicate_items[1:]:
    #         select_items = select_items[3:]
    #         all_select_items.extend(select_items)
    #         all_where_items.extend(where_items)
    #         all_used_variables |= used_variables

    #         join_items = [f'{first_table_name}.trace_id = {table_name}.trace_id']
    #         for var in used_variables:
    #             if var in used_var_to_first_table_name:
    #                 join_items.append(f'{used_var_to_first_table_name[var]} = {select_dict[var]}')
    #             else:
    #                 used_var_to_first_table_name[var] = table_name

    #         join_clauses.append(f'{join_type} data {table_name} ON ({" AND ".join(join_items)})')

    #     full_select_clause = ', '.join(all_select_items)
    #     full_join_clause = '\n'.join(join_clauses)
    #     full_where_clause = ' AND '.join(all_where_items)

    #     query = f"SELECT {full_select_clause} FROM data {first_table_name}\n{full_join_clause}\nWHERE {full_where_clause}"
    #     return query, all_used_variables

    def _build_query_from_predicate_items(self, predicate_items: typing.List[typing.Tuple[typing.List[str], typing.Dict[str, str], typing.List[str], typing.Set[str], str]],
                                          conjunction: bool = True, negation: bool = False) -> typing.Tuple[str, typing.Set[str]]:

        if len(predicate_items) == 1:
            select_items, select_dict, where_items, used_variables, table_name = predicate_items[0]
            return self._build_simple_predicate_query(select_items, where_items, table_name), used_variables

        intervals_logical_op = ' & ' if conjunction else ' | '
        join_type = 'FULL JOIN' if (negation or not conjunction) else 'INNER JOIN'
        first_select_items, first_select_dict, first_where_items, all_used_variables, first_table_name = predicate_items[0]

        full_select_items = []

        # TODO: if disjuncting, coalesce shared variable column
        variable_to_referencing_tables = defaultdict(list)

        # replace the intervals with the conjunction of all intervals
        for col_index, shared_col in enumerate(['trace_id', 'domain']):
            shared_col_per_table = [item[0][col_index] for item in predicate_items]
            full_select_items.append(f'COALESCE({", ".join(shared_col_per_table)}) AS "{shared_col}"')

        itervals_column_names = [f'intervals_{item[-1]}' for item in predicate_items]
        intervals_select_items = [f'{item[0][2]} AS {column_name}' for (item, column_name) in zip(predicate_items, itervals_column_names)]

        for v in all_used_variables:
            variable_to_referencing_tables[v].append(first_select_dict[v])

        first_select_items_no_as = [item.split(' AS ')[0] for item in first_select_items]
        join_clauses = [f'FROM (SELECT {", ".join(first_select_items_no_as)} FROM data {first_table_name} WHERE {" AND ".join(first_where_items)}) {first_table_name}']

        previous_table_names = [first_table_name]

        for select_items, select_dict, where_items, used_variables, table_name in predicate_items[1:]:
            all_used_variables |= used_variables

            join_items = [f'{previous_table_name}.trace_id = {table_name}.trace_id' for previous_table_name in previous_table_names]
            for var in used_variables:

                if var in variable_to_referencing_tables:
                    j = '(' + ' OR '.join(f'{previous_select} = {select_dict[var]}' for previous_select in variable_to_referencing_tables[var]) + ')'
                    join_items.append(j)

                variable_to_referencing_tables[var].append(select_dict[var])

            select_items_no_as = [item.split(' AS ')[0] for item in select_items]
            join_clauses.append(f'{join_type} (SELECT {", ".join(select_items_no_as)} FROM data {table_name} WHERE {" AND ".join(where_items)}) {table_name} ON ({" AND ".join(join_items)})')

            previous_table_names.append(table_name)

        variable_select_items = [f'{referencing_tables[0]} AS "{var}"'
                                if len(referencing_tables) == 1
                                else f'COALESCE({", ".join(referencing_tables)}) AS "{var}"'
                                for var, referencing_tables in variable_to_referencing_tables.items()]

        full_select_clause = ', '.join(full_select_items + variable_select_items + intervals_select_items)
        full_join_clause = '\n'.join(join_clauses)

        inner_query = f"SELECT {full_select_clause}\n{full_join_clause}"

        combined_intervals = intervals_logical_op.join(f'COALESCE({column_name}, {"~" if conjunction else ""}tld.intervals)' for column_name in itervals_column_names)
        combined_intervals = f'({combined_intervals}) AS intervals'

        query = f"""
SELECT predicates.*, {combined_intervals} FROM ({inner_query}) predicates
JOIN trace_length_and_domains tld
ON predicates.trace_id = tld.trace_id
"""
        return query, all_used_variables

    def _build_missing_negated_predicates_query(self, predicate_used_variables: typing.Set[str], negated_used_variables: typing.Set[str],
                                                mapping: typing.Dict[str, typing.Union[str, typing.List[str]]],
                                                predicate_table_name: str = PREDICATE_TABLE_NAME,
                                                negated_predicate_table_name: str = NEGATED_PREDICATE_TABLE_NAME,
                                                conjunction: bool = True) -> str:
        # If it's a disjunction, we anchor based on all possible traces, not just the ones that have the predicate
        if not conjunction:
            predicate_table_name = 'trace_length_and_domains'

        all_used_variables = predicate_used_variables | negated_used_variables
        join_clauses = []
        negated_predicate_where_clauses = [f'{predicate_table_name}.trace_id = {negated_predicate_table_name}.trace_id']
        negated_predicate_full_trace_where_clauses = []

        select_items = [f'{predicate_table_name}.trace_id as trace_id']

        for var_index, var in enumerate(all_used_variables):
            table_name = f'o{var_index}'
            join_items = [f'{predicate_table_name}.domain = {table_name}.domain']
            var_type_list = self._map_variable_to_type_list(var, mapping)
            join_items.append(self._build_missing_object_assignment_where(var_type_list, table_name=table_name))

            # Only anchor on object satisfying the predicate in case of a conjunction
            in_predicate_and_conjunction = var in predicate_used_variables and conjunction
            if in_predicate_and_conjunction:
                join_items.append(f'{table_name}.object_id = {predicate_table_name}."{var}"')

            if var in negated_used_variables:
                if in_predicate_and_conjunction:
                    negated_predicate_where_clauses.append(f'({table_name}.object_id = {negated_predicate_table_name}."{var}" OR {negated_predicate_table_name}."{var}" IS NULL)')

                else:
                    negated_predicate_where_clauses.append(f'{table_name}.object_id = {negated_predicate_table_name}."{var}"')

                if conjunction and not any(arg_type in ALL_NON_OBJECT_TYPES for arg_type in var_type_list):
                    negated_predicate_full_trace_where_clauses.append(f'{table_name}.object_id = {negated_predicate_table_name}."{var}"')

            if DEBUG_MISSING_PREDICATE_QUERY:
                select_items.append(f'{table_name}.object_id AS "{var}"')

            join_clauses.append(f'JOIN object_type_to_id {table_name} ON {" AND ".join(join_items)}')

        inner_where_clause = f'WHERE {" AND ".join(negated_predicate_where_clauses)}'
        where_clause = f'WHERE NOT EXISTS (SELECT 1 FROM negated_predicate {inner_where_clause})'

        # Check for any negated predicates covering the full trace, and exclude them
        if conjunction:
            # Make sure at least one object doesn't match existing negated predicates, from object types
            full_cover_check_where_clauses = [f'{predicate_table_name}.trace_id = {negated_predicate_table_name}.trace_id',
                                              'bit_count(~negated_predicate.intervals) = 0']

            if negated_predicate_full_trace_where_clauses:
                full_cover_check_where_clauses.insert(1, f'({" OR ".join(negated_predicate_full_trace_where_clauses)})')

            where_clause += f'AND NOT EXISTS (SELECT 1 FROM negated_predicate WHERE {" AND ".join(full_cover_check_where_clauses)})'

        full_select_clause = ', '.join(select_items)
        full_join_clause = '\n'.join(join_clauses)
        return f"SELECT {full_select_clause} FROM {predicate_table_name}\n{full_join_clause}\n{where_clause}"

    def _handle_and_refactored(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]],
                               predicate_table_name: str = PREDICATE_TABLE_NAME,
                               negated_predicate_table_name: str = NEGATED_PREDICATE_TABLE_NAME,
                               missing_negated_predicate_table_name: str = MISSING_NEGATED_PREDICATE_TABLE_NAME,
                               return_trace_ids: bool = False,
                               **kwargs) -> typing.Tuple[str, typing.Set[str], bool]:
        and_args = predicate["and_args"]
        if not isinstance(and_args, list):
            and_args = [and_args]

        if len(and_args) > self.max_child_args:
            raise PredicateNotImplementedException("Too many and args")

        predicate_items = []
        negated_predicate_items = []

        for i, arg in enumerate(and_args):
            if arg.parseinfo.rule == 'super_predicate':  # type: ignore
                arg = arg.pred  # type: ignore

            negated = False

            if arg.parseinfo.rule == 'super_predicate_not':  # type: ignore
                negated = True
                arg = arg.not_args.pred  # type: ignore

            if arg.parseinfo.rule == 'predicate':  # type: ignore
                table_name = f'd{i}'
                result = self._build_predicate_select_where_items(arg, mapping, table_name=table_name, **kwargs)  # type: ignore
                result = (*result, table_name)
                if negated:
                    negated_predicate_items.append(result)
                else:
                    predicate_items.append(result)

            else:
                raise PredicateNotImplementedException(f"AND sub-predicate not implemented: {arg.parseinfo.rule}")  # type: ignore

        if len(predicate_items) > 0:
            predicate_query, predicate_used_variables = self._build_query_from_predicate_items(predicate_items, conjunction=True)

        else:
            predicate_query, predicate_used_variables = None, set()

        if len(negated_predicate_items) > 0:
            negated_query, negated_used_variables = self._build_query_from_predicate_items(negated_predicate_items, conjunction=False, negation=True)
            if predicate_query is None:
                predicate_table_name = 'trace_length_and_domains'
            missing_negative_query = self._build_missing_negated_predicates_query(predicate_used_variables, negated_used_variables,
                                                                                  mapping, predicate_table_name=predicate_table_name,
                                                                                  conjunction=True)

        else:
            negated_query, negated_used_variables = None, set()
            missing_negative_query = None

        all_used_variables = predicate_used_variables | negated_used_variables
        shared_variables = predicate_used_variables & negated_used_variables

        select_operator = f'SELECT {predicate_table_name}.trace_id'
        limit_statement = 'LIMIT 1' if not return_trace_ids else ''

        if return_trace_ids:
            select_operator = f'SELECT DISTINCT({predicate_table_name}.trace_id)'

        if negated_query is None:
            return f"{select_operator} FROM ({predicate_query}) {predicate_table_name} WHERE bit_count(intervals) > 0", all_used_variables, True

        negated_predicate_variable_clauses = [f'({predicate_table_name}."{v}" = {negated_predicate_table_name}."{v}" OR {negated_predicate_table_name}."{v}" IS NULL)' for v in shared_variables]
        for negated_only_variable in negated_used_variables - predicate_used_variables:
            negated_predicate_variable_clauses.append(f'{negated_predicate_table_name}."{negated_only_variable}" IS NOT NULL')
        negated_predicate_variable_clauses_where = ' AND '.join(negated_predicate_variable_clauses) + '\n'

        if predicate_query is None:
            negated_select = select_operator.replace(predicate_table_name, negated_predicate_table_name)
            missing_negated_select = select_operator.replace(predicate_table_name, missing_negated_predicate_table_name)

            query = f"""WITH {negated_predicate_table_name} as ({negated_query}),
{missing_negated_predicate_table_name} as ({missing_negative_query})
(
    {missing_negated_select} FROM {missing_negated_predicate_table_name}
    {limit_statement}
) UNION (
    {negated_select} FROM {negated_predicate_table_name}
    WHERE bit_count(~{negated_predicate_table_name}.intervals) > 0
    {'AND' if len(negated_predicate_variable_clauses) > 0 else ''} {negated_predicate_variable_clauses_where}
    {limit_statement}
)
""".strip()

    # JOIN trace_length_and_domains ON trace_length_and_domains.trace_id = {negated_predicate_table_name}.trace_id
    # WHERE bit_count({negated_predicate_table_name}.intervals) < trace_length_and_domains.length

            return query, all_used_variables, True

        query = f"""WITH {predicate_table_name} as ({predicate_query}),
{negated_predicate_table_name} as ({negated_query}),
{missing_negated_predicate_table_name} as ({missing_negative_query})
{select_operator} FROM {predicate_table_name}
WHERE bit_count({predicate_table_name}.intervals) > 0 AND (
EXISTS (
    SELECT * FROM {negated_predicate_table_name}
    WHERE
        {predicate_table_name}.trace_id = {negated_predicate_table_name}.trace_id
        {'AND' if len(negated_predicate_variable_clauses) > 0 else ''} {negated_predicate_variable_clauses_where} AND bit_count({predicate_table_name}.intervals & ~{negated_predicate_table_name}.intervals) > 0
) OR EXISTS (
    SELECT * FROM {missing_negated_predicate_table_name}
    WHERE {predicate_table_name}.trace_id = {missing_negated_predicate_table_name}.trace_id
)
)
""".strip()

        # bit_count({predicate_table_name}.intervals) > bit_count({predicate_table_name}.intervals & {negated_predicate_table_name}.intervals)
        return query, all_used_variables, True

    def _handle_or_refactored(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]],
                               predicate_table_name: str = PREDICATE_TABLE_NAME,
                               negated_predicate_table_name: str = NEGATED_PREDICATE_TABLE_NAME,
                               missing_negated_predicate_table_name: str = MISSING_NEGATED_PREDICATE_TABLE_NAME,
                               return_trace_ids: bool = False,
                               **kwargs) -> typing.Tuple[str, typing.Set[str], bool]:
        or_args = predicate["or_args"]
        if not isinstance(or_args, list):
            or_args = [or_args]

        if len(or_args) > self.max_child_args:
            raise PredicateNotImplementedException("Too many and args")

        predicate_items = []
        negated_predicate_items = []

        for i, arg in enumerate(or_args):
            if arg.parseinfo.rule == 'super_predicate':  # type: ignore
                arg = arg.pred  # type: ignore

            negated = False

            if arg.parseinfo.rule == 'super_predicate_not':  # type: ignore
                negated = True
                arg = arg.not_args.pred  # type: ignore

            if arg.parseinfo.rule == 'predicate':  # type: ignore
                table_name = f'd{i}'
                result = self._build_predicate_select_where_items(arg, mapping, table_name=table_name, **kwargs)  # type: ignore
                result = (*result, table_name)
                if negated:
                    negated_predicate_items.append(result)
                else:
                    predicate_items.append(result)

            else:
                raise PredicateNotImplementedException(f"AND sub-predicate not implemented: {arg.parseinfo.rule}")  # type: ignore

        if len(predicate_items) > 0:
            predicate_query, predicate_used_variables = self._build_query_from_predicate_items(predicate_items, conjunction=False)

        else:
            predicate_query, predicate_used_variables = None, set()

        if len(negated_predicate_items) > 0:
            negated_query, negated_used_variables = self._build_query_from_predicate_items(negated_predicate_items, conjunction=True, negation=True)
            missing_negative_query = self._build_missing_negated_predicates_query(predicate_used_variables, negated_used_variables,
                                                                                  mapping, conjunction=False,)

        else:
            negated_query, negated_used_variables = None, set()
            missing_negative_query = None

        all_used_variables = predicate_used_variables | negated_used_variables
        # shared_variables = predicate_used_variables & negated_used_variables

        select_operator = f'SELECT {predicate_table_name}.trace_id'
        if return_trace_ids:
            select_operator = f'SELECT DISTINCT({predicate_table_name}.trace_id)'

        limit_statement = 'LIMIT 1' if not return_trace_ids else ''
        negated_select = select_operator.replace(predicate_table_name, negated_predicate_table_name)
        missing_negated_select = select_operator.replace(predicate_table_name, missing_negated_predicate_table_name)


        if negated_query is None:
            return f"{select_operator} FROM ({predicate_query}) {predicate_table_name} WHERE bit_count(intervals) > 0", all_used_variables, True

        negated_only_variables = negated_used_variables - predicate_used_variables
        negated_predicate_variable_clauses = [
            f'{negated_predicate_table_name}."{negated_only_variable}" IS NOT NULL'
            for negated_only_variable in negated_only_variables
        ]

        negated_predicate_variable_clauses_where = ' AND '.join(negated_predicate_variable_clauses)

        if predicate_query is None:
            query = f"""WITH {negated_predicate_table_name} as ({negated_query}),
{missing_negated_predicate_table_name} as ({missing_negative_query})
(
    {missing_negated_select} FROM {missing_negated_predicate_table_name}
    {limit_statement}
) UNION (
    {negated_select} FROM {negated_predicate_table_name}
    WHERE bit_count(~{negated_predicate_table_name}.intervals) > 0
    {'AND' if len(negated_predicate_variable_clauses) > 0 else ''} {negated_predicate_variable_clauses_where}
    {limit_statement}
)
""".strip()
            # JOIN trace_length_and_domains ON trace_length_and_domains.trace_id = {negated_predicate_table_name}.trace_id
            # WHERE bit_count({negated_predicate_table_name}.intervals) < trace_length_and_domains.length

            return query, all_used_variables, True

        # shared_variables_where = ' AND '.join([f'({predicate_table_name}."{v}" = {negated_predicate_table_name}."{v}" OR {negated_predicate_table_name}."{v}" IS NULL)' for v in shared_variables])
        find_negated_only_variable_assignments = ''
        if negated_only_variables:
            find_negated_only_variable_assignments = f"""
AND (
EXISTS (
    SELECT 1 FROM {negated_predicate_table_name}
    WHERE {predicate_table_name}.trace_id = {negated_predicate_table_name}.trace_id
    AND {negated_predicate_variable_clauses_where}
    )
OR EXISTS (
    SELECT 1 FROM {missing_negated_predicate_table_name}
    WHERE {predicate_table_name}.trace_id = {missing_negated_predicate_table_name}.trace_id
    AND {negated_predicate_variable_clauses_where.replace(negated_predicate_table_name, missing_negated_predicate_table_name)}
    )
)
""".strip()

        query = f"""
WITH {predicate_table_name} as ({predicate_query}),
{negated_predicate_table_name} as ({negated_query}),
{missing_negated_predicate_table_name} as ({missing_negative_query})
(
    {missing_negated_select} FROM {missing_negated_predicate_table_name}
    {limit_statement}
) UNION  (
    {select_operator} FROM {predicate_table_name}
    WHERE bit_count({predicate_table_name}.intervals) > 0
    {find_negated_only_variable_assignments}
    {limit_statement}
) UNION (
    {negated_select} FROM {negated_predicate_table_name}
    WHERE bit_count({negated_predicate_table_name}.intervals) > 0
    {'AND' if len(negated_predicate_variable_clauses) > 0 else ''} {negated_predicate_variable_clauses_where}
    {limit_statement}
)
""".strip()
        return query, all_used_variables, True

    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_and(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str], bool]:
        and_args = predicate["and_args"]
        if not isinstance(and_args, list):
            and_args = [and_args]

        if len(and_args) > self.max_child_args:
            raise PredicateNotImplementedException("Too many and args")

        sub_queries = []
        used_variables_by_child = []
        all_used_variables = set()

        for and_arg in and_args:  # type: ignore
            try:
                sub_query, sub_used_variables, _ = self._inner_filter(typing.cast(tatsu.ast.AST, and_arg), mapping, **kwargs)
                sub_queries.append(sub_query)
                used_variables_by_child.append(sub_used_variables)
                all_used_variables |= sub_used_variables

            except PredicateNotImplementedException as e:
                continue

        if len(sub_queries) == 0:
            raise PredicateNotImplementedException("All sub-predicates of the and were not implemented")

        if len(sub_queries) == 1:
            return sub_queries[0], used_variables_by_child[0], False

        subquery_table_names = [self._next_temp_table_name() for _ in range(len(sub_queries))]
        with_items = [f"{table_name} AS ({subquery})" for table_name, subquery in zip(subquery_table_names, sub_queries)]

        select_items = [f"{subquery_table_names[0]}.trace_id", f"{subquery_table_names[0]}.domain"]
        selected_variables = set()
        intervals = []
        join_clauses = []

        for i, (table_name, sub_used_variables) in enumerate(zip(subquery_table_names, used_variables_by_child)):
            intervals.append(f"{table_name}.intervals")

            for variable in sub_used_variables:
                if variable not in selected_variables:
                    select_items.append(f'{table_name}."{variable}"')
                    selected_variables.add(variable)

            if i > 0:
                join_parts = [f"INNER JOIN {table_name} ON ({subquery_table_names[0]}.trace_id={table_name}.trace_id)"]
                joined_variables = set()

                for prev_table_name, prev_used_variables in zip(subquery_table_names[:i], used_variables_by_child[:i]):
                    shared_variables = sub_used_variables & prev_used_variables - joined_variables
                    join_parts.extend([f'({table_name}."{v}"={prev_table_name}."{v}")' for v in shared_variables])
                    joined_variables |= shared_variables

                join_clauses.append(" AND ".join(join_parts))


        select_items.append(f'({" & ".join(intervals)}) AS intervals')

        inner_query = f"WITH {', '.join(with_items)} SELECT {', '.join(select_items)} FROM {subquery_table_names[0]} {' '.join(join_clauses)}"

        table_name = self._next_temp_table_name()
        query = f"WITH {table_name} AS ({inner_query}) SELECT * FROM {table_name} WHERE bit_count(intervals) != 0"
        if DEBUG: print(query)
        return query, all_used_variables, False

    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_and_de_morgans(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str], bool]:
        and_args = predicate["and_args"]
        if not isinstance(and_args, list):
            and_args = [and_args]

        if len(and_args) > self.max_child_args:
            raise PredicateNotImplementedException("Too many and args")

        sub_queries = []
        used_variables_by_child = []
        all_used_variables = set()

        for and_arg in and_args:  # type: ignore
            try:
                subquery, sub_used_variables, _ = self._inner_filter(typing.cast(tatsu.ast.AST, and_arg), mapping, **kwargs)
                sub_queries.append(subquery)
                used_variables_by_child.append(sub_used_variables)
                all_used_variables |= sub_used_variables

            except PredicateNotImplementedException as e:
                continue

        if len(sub_queries) == 0:
            raise PredicateNotImplementedException("All sub-predicates of the or were not implemented")

        if len(sub_queries) == 1:
            return sub_queries[0], used_variables_by_child[0], False

        sub_queries.insert(0, self._build_potential_missing_values_query(mapping, list(all_used_variables)))
        used_variables_by_child.insert(0, all_used_variables)

        subquery_table_names = [self._next_temp_table_name() for _ in range(len(sub_queries))]

        with_items = [f"{table_name} AS ({subquery})" for table_name, subquery in zip(subquery_table_names, sub_queries)]

        select_items = [f"{subquery_table_names[0]}.trace_id", f"{subquery_table_names[0]}.domain"]
        selected_variables = set()
        intervals = []
        join_clauses = []

        for i, (sub_table_name, sub_used_variables) in enumerate(zip(subquery_table_names, used_variables_by_child)):
            intervals.append(f"{sub_table_name}.intervals")

            for variable in sub_used_variables:
                if variable not in selected_variables:
                    select_items.append(f'{sub_table_name}."{variable}"')
                    selected_variables.add(variable)

            if i > 0:
                join_parts = [f"LEFT JOIN {sub_table_name} ON ({subquery_table_names[0]}.trace_id={sub_table_name}.trace_id)"]

                shared_variables = sub_used_variables & all_used_variables
                join_parts.extend([f'({subquery_table_names[0]}."{v}"={sub_table_name}."{v}")' for v in shared_variables])

                join_clauses.append(" AND ".join(join_parts))

        intervals_coalesce = [f"~{intervals_select}" if i > 0 else intervals_select for i, intervals_select in enumerate(intervals)]
        select_items.append(f'~({" | ".join(intervals_coalesce)}) AS intervals')

        inner_query = f"WITH {', '.join(with_items)} SELECT {', '.join(select_items)} FROM {subquery_table_names[0]} {' '.join(join_clauses)}"

        table_name = self._next_temp_table_name()
        query = f"WITH {table_name} AS ({inner_query}) SELECT * FROM {table_name} WHERE intervals IS NOT NULL AND (bit_count(intervals) != 0)"
        if DEBUG: print(query)
        return query, all_used_variables, False

    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_or(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str], bool]:
        or_args = predicate["or_args"]
        if not isinstance(or_args, list):
            or_args = [or_args]

        if len(or_args) > self.max_child_args:
            raise PredicateNotImplementedException("Too many and args")

        sub_queries = []
        used_variables_by_child = []
        all_used_variables = set()

        for or_arg in or_args:  # type: ignore
            try:
                subquery, sub_used_variables, _ = self._inner_filter(typing.cast(tatsu.ast.AST, or_arg), mapping, or_argument=True, **kwargs)
                sub_queries.append(subquery)
                used_variables_by_child.append(sub_used_variables)
                all_used_variables |= sub_used_variables

            except PredicateNotImplementedException as e:
                continue

        if len(sub_queries) == 0:
            raise PredicateNotImplementedException("All sub-predicates of the or were not implemented")

        if len(sub_queries) == 1:
            table_name = self._next_temp_table_name()
            query = f"WITH {table_name} AS ({sub_queries[0]}) SELECT * FROM {table_name} WHERE bit_count(intervals) != 0"
            return query, used_variables_by_child[0], False

        # Trying to remove this since I only handle one OR/AND at a time
        # Can't actually remove it -- it causes some queries to fail
        # sub_queries.insert(0, self._build_potential_missing_values_query(mapping, list(all_used_variables)))
        # used_variables_by_child.insert(0, all_used_variables)

        subquery_table_names = [self._next_temp_table_name() for _ in range(len(sub_queries))]

        with_items = [f"{table_name} AS ({subquery})" for table_name, subquery in zip(subquery_table_names, sub_queries)]

        select_items = [f"{subquery_table_names[0]}.trace_id", f"{subquery_table_names[0]}.domain"]
        selected_variables = set()
        intervals = []
        join_clauses = []
        first_table_name_by_variable = {}

        for i, (sub_table_name, sub_used_variables) in enumerate(zip(subquery_table_names, used_variables_by_child)):
            sub_table_intervals = f"{sub_table_name}_intervals"
            select_items.append(f"{sub_table_name}.intervals AS {sub_table_intervals}")
            intervals.append(sub_table_intervals)

            for var in sub_used_variables:
                if var not in selected_variables:
                    select_items.append(f'{sub_table_name}."{var}"')
                    selected_variables.add(var)
                    first_table_name_by_variable[var] = sub_table_name

            if i > 0:
                # join_parts = [f"LEFT JOIN {sub_table_name} ON ({subquery_table_names[0]}.trace_id={sub_table_name}.trace_id)"]
                join_parts = [f"FULL JOIN {sub_table_name} ON ({subquery_table_names[0]}.trace_id={sub_table_name}.trace_id)"]

                for var in sub_used_variables:
                    if first_table_name_by_variable[var] != sub_table_name:
                        join_parts.append(f'({first_table_name_by_variable[var]}."{var}"={sub_table_name}."{var}")')

                # I think the below is an empty statement since all_used_variables is a conjunction of all sub_used_variables
                # shared_variables = sub_used_variables & all_used_variables
                # join_parts.extend([f'({subquery_table_names[0]}."{v}"={sub_table_name}."{v}")' for v in shared_variables])

                join_clauses.append(" AND ".join(join_parts))

        # intervals_coalesce = [f"COALESCE({intervals_select}, {intervals[0]})" if i > 0 else intervals_select for i, intervals_select in enumerate(intervals)]
        # select_items.append(f'({" | ".join(intervals_coalesce)}) AS intervals')
        # select_items.append(f'({" | ".join(intervals)}) AS intervals')

        inner_query = f"WITH {', '.join(with_items)} SELECT {', '.join(select_items)} FROM {subquery_table_names[0]} {' '.join(join_clauses)}"
        table_name = self._next_temp_table_name()

        select_items_without_intervals = [f'{table_name}.{item.split(".")[1]}' for item in select_items if not item.endswith('intervals')]
        intervals_coalesce = [f"COALESCE({intervals_column}, tld.intervals)" for intervals_column in intervals]
        intervals_select = f'({" | ".join(intervals_coalesce)}) AS intervals'
        select_items_without_intervals.append(intervals_select)

        query = f"""
WITH {table_name} AS ({inner_query})
SELECT {', '.join(select_items_without_intervals)} FROM {table_name}
JOIN trace_length_and_domains tld
ON {table_name}.trace_id = tld.trace_id
"""
        # WHERE bit_count({intervals_select}) != 0
        if DEBUG: print(query)
        return query, all_used_variables, False

    def _types_to_arg_casts(self, types: typing.Collection[str]):
        return '(' + ', '.join(f"'{t}'::arg_type" for t in types) + ')'

    def _build_missing_object_assignment_where(self, object_types: typing.Union[str, typing.List[str]], table_name: typing.Optional[str] = None) -> str:
        if table_name is not None:
            table_name = f"{table_name}."
        else:
            table_name = ""

        if isinstance(object_types, str) and object_types in META_TYPES:
            object_types = META_TYPES[object_types]

        else:
            object_types = sum([META_TYPES.get(arg_type, [arg_type]) if arg_type != BLOCK else [BLOCK] for arg_type in object_types], [])

        if isinstance(object_types, str) or len(object_types) == 1:
            if isinstance(object_types, list):
                object_types = object_types[0]

            if object_types == GAME_OBJECT:
                where_clause = f"{table_name}is_game_object = true"
            elif object_types == BLOCK:
                where_clause = f"{table_name}is_block = true"
            else:
                where_clause = f"{table_name}type = '{object_types}'"
        else:
            where_clause_items = []
            where_types = []

            for object_type in object_types:
                if object_type == GAME_OBJECT:
                    where_clause_items.append(f"{table_name}is_game_object = true")

                elif object_type == BLOCK:
                    where_clause_items.append(f"{table_name}is_block = true")

                else:
                    where_types.append(object_type)

            if len(where_types) > 0:
                where_clause_items.append(f"{table_name}type IN {self._types_to_arg_casts(where_types)}")

            if len(where_clause_items) == 1:
                where_clause = where_clause_items[0]

            else:
                where_clause = f"({' OR '.join(where_clause_items)})"

        return where_clause

    def _build_object_assignment_cte(self, var: str, object_types: typing.Union[str, typing.List[str]]):
        where_clause = self._build_missing_object_assignment_where(object_types)
        return f'SELECT domain, object_id AS "{var}" FROM object_type_to_id WHERE {where_clause}'

    def object_assignments_query(self, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]]):
        if len(mapping) == 0:
            return None

        if len(mapping) == 1:
            first_key = list(mapping.keys())[0]
            query = self._build_object_assignment_cte(first_key, mapping[first_key])

        else:
            object_id_selects = []
            ctes = []
            join_statements = []
            variables = list(mapping.keys())
            table_names = [self._next_temp_table_name() for _ in range(len(variables))]
            for i, (var, table_name) in enumerate(zip(variables, table_names)):
                var_types = mapping[var]
                ctes.append(f"{table_name} AS ({self._build_object_assignment_cte(var, var_types)})")
                object_id_selects.append(f'{table_name}."{var}" AS "{var}"')
                if i > 0:
                    join_clauses = []
                    join_clauses.append(f"({table_names[0]}.domain = {table_name}.domain)")
                    for j in range(i):
                        join_clauses.append(f'({table_names[j]}."{variables[j]}" != {table_name}."{var}")')

                    join_statements.append(f"JOIN {table_name} ON {' AND '.join(join_clauses)}")

            with_statement = ',\n'.join(ctes)
            join_statement = '\n'.join(join_statements)

            query = f"""WITH {with_statement}
SELECT {table_names[0]}.domain, {', '.join(object_id_selects)} FROM {table_names[0]}
{join_statement}
"""

        return query

    def _build_potential_missing_values_query(self, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], relevant_vars: typing.List[str]):
        # For each trace ID, and each assignment of the vars that exist in the sub_predicate_df so far:
        relevant_var_mapping = {var: mapping[var] if var.startswith("?") else [var] for var in relevant_vars}

        object_assignments_query = self.object_assignments_query(relevant_var_mapping)

        select_variables = ', '.join(f'object_assignments."{var}" as "{var}"' for var in relevant_vars)
        query = f"SELECT trace_length_and_domains.trace_id AS trace_id, trace_length_and_domains.domain AS domain, trace_length_and_domains.intervals AS intervals, {select_variables} FROM trace_length_and_domains"

        if object_assignments_query is not None:
            query += f" JOIN ({object_assignments_query}) AS object_assignments ON (trace_length_and_domains.domain = object_assignments.domain)"

        if DEBUG: print(query)
        return query

    # @cachetools.cachedmethod(operator.attrgetter('cache'), key=_predicate_and_mapping_cache_key)
    def _handle_not(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str], bool]:
        try:
            inner_query, used_variables, _ = self._inner_filter(typing.cast(tatsu.ast.AST, predicate["not_args"]), mapping, **kwargs)
        except PredicateNotImplementedException as e:
            raise PredicateNotImplementedException(f"Sub-predicate of the not ({e.args}) was not implemented")

        relevant_vars = list(used_variables)
        potential_missing_values_query = self._build_potential_missing_values_query(mapping, relevant_vars)
        potential_missing_values_table_name = self._next_temp_table_name()
        inner_table_name = self._next_temp_table_name()

        # Now, for each possible combination of args on each trace / domain, 'intervals' will contain the truth intervals if
        # they exist and null otherwise, and 'intervals_right' will contain the empty interval'
        join_columns = ["trace_id"] + relevant_vars

        select_items = [f"{potential_missing_values_table_name}.trace_id as trace_id", f"{potential_missing_values_table_name}.domain as domain"]
        select_items.extend(f'{potential_missing_values_table_name}."{var}" as "{var}"' for var in relevant_vars)
        select_items.append(f"(~( {potential_missing_values_table_name}.intervals | COALESCE({inner_table_name}.intervals, {potential_missing_values_table_name}.intervals) )) AS intervals")

        join_items = [f'{potential_missing_values_table_name}."{column}"={inner_table_name}."{column}"'  for column in join_columns]

        not_query = f"""WITH {potential_missing_values_table_name} AS ({potential_missing_values_query}), {inner_table_name} AS ({inner_query})
        SELECT {', '.join(select_items)} FROM {potential_missing_values_table_name} LEFT JOIN {inner_table_name} ON {' AND '.join(join_items)}
        """

        if 'or_argument' in kwargs and kwargs['or_argument']:
            return not_query, used_variables, False

        table_name = self._next_temp_table_name()
        query = f"WITH {table_name} AS ({not_query}) SELECT * FROM {table_name} WHERE bit_count(intervals) != 0"
        if DEBUG: print(query)
        return query, used_variables, False

    def _inner_filter(self, predicate: tatsu.ast.AST, mapping: typing.Dict[str, typing.Union[str, typing.List[str]]], **kwargs) -> typing.Tuple[str, typing.Set[str], bool]:
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
            if 'use_refactored_impl' in kwargs and kwargs['use_refactored_impl']:
                return self._handle_and_refactored(predicate, mapping, **kwargs)

            if 'use_de_morgans' in kwargs and kwargs['use_de_morgans']:
                return self._handle_and_de_morgans(predicate, mapping, **kwargs)

            return self._handle_and(predicate, mapping, **kwargs)

        elif predicate_rule == "super_predicate_or":
            if 'use_refactored_impl' in kwargs and kwargs['use_refactored_impl']:
                return self._handle_or_refactored(predicate, mapping, **kwargs)

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

    stats = CommonSensePredicateStatisticsFullDatabse(cache_dir=DEFAULT_CACHE_DIR,
                                                    # trace_names=CURRENT_TEST_TRACE_NAMES,
                                                    trace_names=FULL_PARTICIPANT_TRACE_SET,
                                                    cache_rules=[],
                                                    base_trace_path=DEFAULT_BASE_TRACE_PATH,
                                                    force_trace_names_hash='028b3733',
                                                    # overwrite=True
                                                    )

    variable_type_usage = json.loads(open(f"{get_project_dir()}/reward-machine/caches/variable_type_usage.json", "r").read())
    for var_type in variable_type_usage:
        if var_type in META_TYPES:
            continue

        n_intervals = stats.con.execute(f"SELECT count(*) FROM data WHERE (arg_1_type='{var_type}' OR arg_2_type='{var_type}');").fetchone()[0]  # type: ignore

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
