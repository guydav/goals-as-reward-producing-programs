from argparse import Namespace
import io
import gc
import os
import numpy as np
import pstats
import tatsu
import time
import tracemalloc
import tqdm
import sys


sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))

from src import fitness_energy_utils as utils
from src.fitness_energy_utils import NON_FEATURE_COLUMNS
from src.ast_counter_sampler import *
from src.ast_utils import cached_load_and_parse_games_from_file, load_games_from_file, _extract_game_id
from src import ast_printer
from src.fitness_features import *


PROFILE = False
MEMORY_TRACE = False
N_REPEATS = 10


if __name__ == '__main__':
    grammar = open('./dsl/dsl.ebnf').read()
    grammar_parser = tatsu.compile(grammar)
    game_asts = list(cached_load_and_parse_games_from_file('./dsl/interactive-beta.pddl', grammar_parser, False, relative_path='.'))

    if MEMORY_TRACE:
        tracemalloc.start()

    args = Namespace(
        no_binarize=False,
        no_merge=False,
        use_specific_objects_ngram_model=False,
        include_arg_types_terms=False,
        include_predicate_under_modal_terms=False,
        include_compositionality_terms=False,
    )
    featurizer = build_fitness_featurizer(args)
    np.seterr(all='raise')

    profile = None
    if PROFILE:
        import cProfile
        profile = cProfile.Profile()
        profile.enable()

    if MEMORY_TRACE:
        tracemalloc.start(25)
        size, peak = tracemalloc.get_traced_memory()
        print(f'After creating featurizer | Memory usage is {size / 10**6}MB | Peak was {peak / 10**6}MB')
        tracemalloc.reset_peak()

    timings = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        for i in range(0, len(game_asts)):
        # for i in range(53, 54):
        # for i in tqdm(range(0, len(game_asts))):
            # first_snapshot = tracemalloc.take_snapshot()

            # print(f'Parsing game #{i}: {game_asts[i][1].game_name}')
            featurizer.parse(game_asts[i], 'interactive-beta.pddl')

            # if MEMORY_TRACE:
            #     # gc.collect()
            #     size, peak = tracemalloc.get_traced_memory()
            #     print(f'After current game | Memory usage is {size / 10**6}MB | Peak was {peak / 10**6}MB')
            #     tracemalloc.reset_peak()
            #     print()

            # second_snapshot = tracemalloc.take_snapshot()
            # top_stats = second_snapshot.compare_to(first_snapshot, 'lineno')
            # print("[ Top 10 differences ]")
            # for stat in top_stats[:10]:
            #     print(stat)


        end = time.perf_counter()
        timings.append(end - start)

    print(f'\nParsing took an average of {np.mean(timings):.4f} +/- {np.std(timings):.4f} seconds (n={N_REPEATS}))')

    if MEMORY_TRACE:
        size, peak = tracemalloc.get_traced_memory()
        print(f'After finishing to parse | Memory usage is {size / 10**6}MB | Peak was {peak / 10**6}MB')
        tracemalloc.reset_peak()

        # print('=' * 100)

        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('traceback')
        # # pick the biggest memory block
        # for block_index in range(5):
        #     stat = top_stats[block_index]
        #     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        #     for line in stat.traceback.format():
        #         print(line)

        #     print('=' * 100)

    if profile is not None:
        profile.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    d = featurizer.to_df()
    print(d[[c for c in d.columns if 'predicate_found_in_data_' in c]].describe())
