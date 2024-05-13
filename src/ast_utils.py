import argparse
from collections import namedtuple, deque
import copy
from itertools import chain
import enum
import gzip
import hashlib
import logging
import multiprocess as multiprocessing
import numpy as np
import os
import pickle
import tatsu
import tatsu.ast
import tatsu.infos
import tatsu.grammars
import tempfile
import tqdm
import typing
import sys

import msgpack

import ast_printer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DEFAULT_TEST_FILES = (
    './problems-few-objects.pddl',
    './problems-medium-objects.pddl',
    './problems-many-objects.pddl'
)


def load_asts(args: argparse.Namespace, grammar_parser: tatsu.grammars.Grammar,
    should_print: bool = False):

    if not args.test_files:
        args.test_files.extend(DEFAULT_TEST_FILES)

    if should_print:
        results = []
        for test_file in args.test_files:
            for game in load_games_from_file(test_file):
                print(game)
                results.append(grammar_parser.parse(game))
        return results

    else:
        return [grammar_parser.parse(game)
            for test_file in args.test_files
            for game in load_games_from_file(test_file)]


DEFAULT_STOP_TOKENS = ('(define', )  # ('(:constraints', )


def load_games_from_file(path: str, start_token: str='(define',
    stop_tokens: typing.Optional[typing.Sequence[str]] = None,
    remove_comments: bool = True, comment_prefixes_to_keep: typing.Optional[typing.Sequence[str]] = None):

    if stop_tokens is None or not stop_tokens:
        stop_tokens = DEFAULT_STOP_TOKENS

    open_method = gzip.open if path.endswith('.gz') else open

    with open_method(path, 'rt') as f:
        lines = f.readlines()
        # new_lines = []
        # for l in lines:
        #     if not l.strip()[0] == ';':
        #         print(l)
        #         new_lines.append(l[:l.find(';')])

        if remove_comments:
            new_lines = [l[:l.find(';')] for l in lines
                if len(l.strip()) > 0 and not l.strip()[0] == ';']

        else:
            new_lines = []
            for l in lines:
                l_s = l.strip()
                if l_s.startswith(';') and (comment_prefixes_to_keep is None or any(l_s.startswith(prefix) for prefix in comment_prefixes_to_keep)):
                    new_lines.append(l.rstrip())
                elif not l_s.startswith(';'):
                    new_lines.append(l[:l.find(';')])

        text = '\n'.join(new_lines)
        start = text.find(start_token)

        while start != -1:
            end_matches = [text.find(stop_token, start + 1) for stop_token in stop_tokens]  # type: ignore
            end_matches = [match != -1 and match or len(text) for match in end_matches]
            end = min(end_matches)
            next_start = text.find(start_token, start + 1)
            if end <= next_start or end == len(text):  # we have a match
                test_case = text[start:end]
                if end < next_start:
                    test_case += ')'

                yield test_case.strip()

            start = next_start




CACHE_FOLDER = os.path.abspath(os.environ.get('GAME_GENERATION_CACHE', os.path.join(tempfile.gettempdir(), 'game_generation_cache')))
logger.debug(f'Using cache folder: {CACHE_FOLDER}')
CACHE_FILE_PATTERN = '{name}-cache.pkl.gz'
CACHE_HASHES_KEY = 'hashes'
CACHE_ASTS_KEY = 'asts'
CACHE_DSL_HASH_KEY = 'dsl'


def _generate_cache_file_name(file_path: str, relative_path: typing.Optional[str] = None):
    if not os.path.exists(CACHE_FOLDER):
        logger.debug(f'Creating cache folder: {CACHE_FOLDER}')
        os.makedirs(CACHE_FOLDER, exist_ok=True)

    name = os.path.basename(file_path).split('.', 1)[0]
    # if relative_path is not None:
    #     return os.path.join(relative_path, CACHE_FOLDER, CACHE_FILE_PATTERN.format(name=name))
    # else:
    return os.path.join(CACHE_FOLDER, CACHE_FILE_PATTERN.format(name=name))


def _extract_game_id(game_str: str):
    start = game_str.find('(game') + 6
    end = game_str.find(')', start)
    full_game_id = game_str[start:end]

    # If it's a long name produced by a regrowth, truncate it at <id>-<index>-<regrowth-index>
    first_dash_index = full_game_id.find('-')
    second_dash_index = full_game_id.find('-', first_dash_index + 1)
    third_dash_index = full_game_id.find('-', second_dash_index + 1)
    if second_dash_index != -1:
        return full_game_id[:third_dash_index]

    return full_game_id


def fixed_hash(str_data: str):
    return hashlib.md5(bytearray(str_data, 'utf-8')).hexdigest()


class NoParseinfoTokenizerModelContext(tatsu.grammars.ModelContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_parseinfo(self, name, pos):
        parseinfo = super()._get_parseinfo(name, pos)
        if parseinfo is not None:
            parseinfo = parseinfo._replace(tokenizer=None)

        return parseinfo


class FullCacheRebuilder:
    def __init__(self, games_file_path: str, grammar_parser: tatsu.grammars.Grammar,
                 use_tqdm: bool, remove_parseinfo_tokenizers: bool = True,
                 expected_total_count: typing.Optional[int] = None,
                 n_workers: int = 15, chunksize: int = 1250):

        self.games_file_path = games_file_path
        self.use_tqdm = use_tqdm
        self.remove_parseinfo_tokenizers = remove_parseinfo_tokenizers
        self.expected_total_count = expected_total_count
        self.n_workers = n_workers
        self.chunksize = chunksize

        self.grammar_parser = copy.deepcopy(grammar_parser)
        self.config = grammar_parser.config.replace_config(None)
        self.context = NoParseinfoTokenizerModelContext(grammar_parser.rules, config=self.config)

        # self._grammar_parsers = [copy.deepcopy(grammar_parser) for _ in range(n_workers)]
        # self._configs = []
        # self._contexts = []

    #     for grammar_parser in self._grammar_parsers:
    #         config = None
    #         context = None
    #         if self.remove_parseinfo_tokenizers:
    #             config = grammar_parser.config.replace_config(None)
    #             context = NoParseinfoTokenizerModelContext(grammar_parser.rules, config=config)

    #         self._configs.append(config)
    #         self._contexts.append(context)

    # def _process_index(self):
    #     identity = multiprocessing.current_process()._identity  # type: ignore
    #     if identity is None or len(identity) == 0:
    #         return 0

    #     return identity[0] % self.n_workers

    # @property
    # def grammar_parser(self):
    #     return self._grammar_parsers[self._process_index()]

    # @property
    # def config(self):
    #     return self._configs[self._process_index()]

    # @property
    # def context(self):
    #     return self._contexts[self._process_index()]

    def _parse_single_game(self, game):
        game_id = _extract_game_id(game)
        game_hash = fixed_hash(game)
        ast = self.grammar_parser.parse(game, config=self.config, ctx=self.context)
        # msgpack_ast_restore(msgpack.unpackb(ast_bytes), copy_type=ASTCopyType.FULL)
        return game_id, game_hash, ast  # msgpack.packb(ast)

    def __call__(self, cache):
        game_iter = load_games_from_file(self.games_file_path)

        multiprocessing.set_start_method('spawn', force=True)  # type: ignore
        with multiprocessing.Pool(self.n_workers) as p:  # type: ignore
            logger.info('Pool started')

            if self.n_workers > 1:
                result_iter = p.imap_unordered(self._parse_single_game, game_iter, chunksize=self.chunksize)

            else:
                result_iter = map(self._parse_single_game, game_iter)

            if self.use_tqdm:
                result_iter = tqdm.tqdm(result_iter, total=self.expected_total_count)

            for game_id, game_hash, ast in result_iter:
                cache[CACHE_HASHES_KEY][game_id] = game_hash
                cache[CACHE_ASTS_KEY][game_id] = ast  # msgpack_ast_restore(msgpack.unpackb(ast_bytes), copy_type=ASTCopyType.FULL)

        logger.info(f'Pool ended, cache size is {len(cache[CACHE_HASHES_KEY])}')
        return cache



def cached_load_and_parse_games_from_file(games_file_path: str, grammar_parser: tatsu.grammars.Grammar,
    use_tqdm: bool, relative_path: typing.Optional[str] = None,
    save_updates_every: int = -1, log_every_change: bool = True,
    remove_parseinfo_tokenizers: bool = True, force_rebuild: bool = False,
    force_from_cache: bool = False,
    full_rebuild_expected_total_count: typing.Optional[int] = None,
    full_rebuild_n_workers: int = 15, full_rebuild_chunksize: int = 1250):

    cache_path = _generate_cache_file_name(games_file_path, relative_path)
    logger.info(f'Loading from cache file: {cache_path}')
    grammar_hash = fixed_hash(grammar_parser._to_str())

    if os.path.exists(cache_path):
        with gzip.open(cache_path, 'rb') as f:
            cache = pickle.load(f)

        logger.info(f'Finished loading cache file: {cache_path}')
    else:
        cache = {CACHE_HASHES_KEY: {}, CACHE_ASTS_KEY: {},
            CACHE_DSL_HASH_KEY: grammar_hash}

        force_rebuild = True

        logger.info(f'No cache file found, creating new cache file for: {cache_path}')
        if force_from_cache:
            logger.warn('Cannot force from cache when there is not cache; setting force_from_cache = False')
            force_from_cache = False

    cache_updates = {CACHE_HASHES_KEY: {}, CACHE_ASTS_KEY: {},
            CACHE_DSL_HASH_KEY: grammar_hash}
    n_cache_updates = 0

    cache_updated = False
    grammar_changed = CACHE_DSL_HASH_KEY not in cache or cache[CACHE_DSL_HASH_KEY] != grammar_hash
    if grammar_changed:
        if CACHE_DSL_HASH_KEY not in cache:
            logger.info('No cached DSL hash found')
        else:
            logger.info('Grammar changed, clearing cache')

        cache[CACHE_DSL_HASH_KEY] = grammar_hash
        cache_updated = True
        force_rebuild = True

    if force_from_cache:
        if grammar_changed:
            raise ValueError('Cannot force from cache when grammar changed')

        for game_id in cache[CACHE_ASTS_KEY]:
            yield cache[CACHE_ASTS_KEY][game_id]

        return

    game_iter = load_games_from_file(games_file_path)
    if use_tqdm:
        game_iter = tqdm.tqdm(game_iter)

    if force_rebuild:
        logger.info('Forcing full rebuild')
        rebuilder = FullCacheRebuilder(
            games_file_path, grammar_parser, use_tqdm,
            remove_parseinfo_tokenizers, full_rebuild_expected_total_count,
            full_rebuild_n_workers, full_rebuild_chunksize)

        cache = rebuilder(cache)

        with gzip.open(cache_path, 'wb') as f:
            pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)

        # Just rebuilt, so no need to check hashes
        for game in game_iter:
            game_id = _extract_game_id(game)
            yield cache[CACHE_ASTS_KEY][game_id]

    else:
        for game in game_iter:
            game_id = _extract_game_id(game)
            game_hash = fixed_hash(game)

            if grammar_changed or game_id not in cache[CACHE_HASHES_KEY] or cache[CACHE_HASHES_KEY][game_id] != game_hash:
                if not grammar_changed and log_every_change:
                    if game_id not in cache[CACHE_HASHES_KEY]:
                        logger.debug(f'Game not found in cache: {game_id}')
                    else:
                        logger.debug(f'Game changed: {game_id}')
                cache_updated = True

                config = None
                ctx = None
                if remove_parseinfo_tokenizers:
                    config = grammar_parser.config.replace_config(None)
                    ctx = NoParseinfoTokenizerModelContext(grammar_parser.rules, config=config)

                ast = grammar_parser.parse(game, config=config, ctx=ctx)
                cache_updates[CACHE_HASHES_KEY][game_id] = game_hash
                cache_updates[CACHE_ASTS_KEY][game_id] = ast
                n_cache_updates += 1

            else:
                ast = cache[CACHE_ASTS_KEY][game_id]

            yield ast

            if save_updates_every > 0 and n_cache_updates >= save_updates_every:
                logger.debug(f'Updating cache with {n_cache_updates} new games')
                cache[CACHE_HASHES_KEY].update(cache_updates[CACHE_HASHES_KEY])
                cache[CACHE_ASTS_KEY].update(cache_updates[CACHE_ASTS_KEY])
                with gzip.open(cache_path, 'wb') as f:
                    pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)
                cache_updates = {CACHE_HASHES_KEY: {}, CACHE_ASTS_KEY: {},
                    CACHE_DSL_HASH_KEY: grammar_hash}
                n_cache_updates = 0

    if n_cache_updates > 0:
        logger.debug(f'Updating cache with {n_cache_updates} new games')
        cache[CACHE_HASHES_KEY].update(cache_updates[CACHE_HASHES_KEY])
        cache[CACHE_ASTS_KEY].update(cache_updates[CACHE_ASTS_KEY])
        cache_updated = True

    if cache_updated:
        logger.debug(f'About to finally update the cache')
        with gzip.open(cache_path, 'wb') as f:
            pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)


def copy_ast(grammar_parser: tatsu.grammars.Grammar, ast: tatsu.ast.AST):
    ast_printer.reset_buffers(True)
    ast_printer.pretty_print_ast(ast)
    ast_str = ''.join(ast_printer.BUFFER)  # type: ignore
    return grammar_parser.parse(ast_str)


def update_ast(ast: tatsu.ast.AST, key: str, value: typing.Any):
    if isinstance(ast, tatsu.ast.AST):
        super(tatsu.ast.AST, ast).__setitem__(key, value)


def apply_selector_list(parent: tatsu.ast.AST, selector: typing.Sequence[typing.Union[str, int]],
    max_index: typing.Optional[int] = None):

    if max_index is None:
        max_index = len(selector)
    for s in selector[:max_index]:
        parent = parent[s]  # type: ignore
    return parent


def replace_child(parent: typing.Union[tuple, tatsu.ast.AST], selector: typing.Union[str, typing.Sequence[typing.Union[str, int]]],
    new_value: typing.Any):

    if isinstance(selector, str):
        selector = (selector,)

    if isinstance(parent, tuple):
        if len(selector) != 1 or not isinstance(selector[0], int):
            raise ValueError('Invalid selector for tuple: {}'.format(selector))

        child_index = selector[0]
        return (*parent[:child_index], new_value, *parent[child_index + 1:])

    last_parent = apply_selector_list(parent, selector, -1)
    last_selector = selector[-1]

    if isinstance(last_selector, str):
        update_ast(last_parent, last_selector, new_value)

    elif isinstance(last_selector, int):
        last_parent[last_selector] = new_value

    else:
        raise ValueError(f'replace_child received last selector of unknown type: {last_selector} ({type(last_selector)})', parent, selector)


def find_all_parents(parent_mapping: typing.Dict[tatsu.infos.ParseInfo, tuple], ast: tatsu.ast.AST):
    parents = []
    parent = parent_mapping[ast.parseinfo][1]  # type: ignore
    while parent is not None and parent != 'root':
        parents.append(parent)
        if isinstance(parent, tuple):
            parent = None
        else:
            parent = parent_mapping[parent.parseinfo][1]

    return parents


def find_selectors_from_root(parent_mapping: typing.Dict[tatsu.infos.ParseInfo, tuple],
    ast: tatsu.ast.AST, root_node: typing.Union[str, tatsu.ast.AST] = 'root'):
    selectors = []
    parent = ast
    while parent != root_node:
        _, parent, selector = parent_mapping[parent.parseinfo]  # type: ignore
        selectors = selector + selectors

    return selectors


def simplified_context_deepcopy(context: dict) -> typing.Dict[str, typing.Union[typing.Dict, typing.Set, int]]:
    context_new = {}

    for k, v in context.items():
        if isinstance(v, dict):
            context_new[k] = dict(v)
        elif isinstance(v, set):
            context_new[k] = set(v)
        elif isinstance(v, (int, float, str, tuple, np.random.Generator)):
            context_new[k] = v
        else:
            raise ValueError(f'Unexpected value type: {v}, {type(v)}')

    return context_new


class ASTCopyType(enum.Enum):
    FULL = 0
    SECTION = 1
    NODE = 2


def msgpack_ast_restore(obj, copy_type: ASTCopyType = ASTCopyType.FULL, depth: int = 0):
    if depth == 0 and copy_type in (ASTCopyType.FULL, ASTCopyType.SECTION):
        return tuple([msgpack_ast_restore(item, copy_type, depth + 1) for item in obj])

    if isinstance(obj, list):
        out = [msgpack_ast_restore(item, copy_type, depth + 1) for item in obj]
        if depth == 1 and copy_type == ASTCopyType.FULL:
            return tuple(out)

        return out

    if isinstance(obj, dict):
        out = {}
        for key, val in obj.items():
            if key == 'parseinfo':
                val = tatsu.infos.ParseInfo(*val)

            out[key] = msgpack_ast_restore(val, copy_type, depth + 1)

        return tatsu.ast.AST(out)

    return obj


T = typing.TypeVar('T')


def deepcopy_ast(ast: T, copy_type: ASTCopyType = ASTCopyType.FULL) -> T:
    # return pickle.loads(pickle.dumps(ast, pickle.HIGHEST_PROTOCOL))
    return msgpack_ast_restore(msgpack.unpackb(msgpack.packb(ast)), copy_type=copy_type)  # type: ignore


def object_total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
