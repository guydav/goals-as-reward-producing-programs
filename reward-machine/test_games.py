import numpy as np
import pathlib
import pytest
import typing
import sys

from utils import FullState, get_project_dir
from manual_run import _load_trace
from game_handler import GameHandler
from preference_handler import PreferenceSatisfaction



BALL_TO_BIN_FROM_BED_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/ball_to_bin_from_bed_trace.json')
BLOCK_STACKING_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/block_stacking_test_trace.json')
BALL_TO_WALL_TO_BIN_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/three_wall_to_bin_bounces.json')
BUILDING_IN_TOUCH_TEST_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/weZ1UVzKNaiTjaqu0DGI-preCreateGame-buildings-in-touching.json')
THREE_WALL_TO_BIN_BOUNCES_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/three_wall_to_bin_bounces.json')
THROW_ALL_DODGEBALLS_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/throw_all_dodgeballs.json')
SETUP_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/setup_test_trace.json')
THROW_BALL_UNIQUE_POSITIONS_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/throw_ball_to_bin_unique_positions.json')
COMPLEX_STACKING_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/complex_stacking_trace.json')

# CLEANUP_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/updated-game-27.json')
CLEANUP_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/qK8hfQE9E97kZMDdL4Hv-preCreateGame-rerecorded.json')
# CLEANUP_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/qK8hfQE9E97kZMDdL4Hv-preCreateGame-rerecorded.json')

TEST_AGENT_DOOR_ADJACENT_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/agent_door_adjacent.json')
# TEST_AGENT_DOOR_ADJACENT_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/sp9Aq6hJRgnebtopwZDN-preCreateGame-rerecorded.json')

# TEST_BLOCK_CACHE_TEST = pathlib.Path(get_project_dir() + '/reward-machine/traces/throw_with_stacked_blocks.json')
# TEST_BLOCK_CACHE_TEST = pathlib.Path(get_project_dir() + '/reward-machine/traces/otcaCEGfUhzEfGy72Qm8-preCreateGame.json')
TEST_BLOCK_CACHE_TEST = pathlib.Path(get_project_dir() + '/reward-machine/traces/otcaCEGfUhzEfGy72Qm8-preCreateGame-rerecorded.json')



def load_game(game_name: str):
    game_path = pathlib.Path(get_project_dir() + f'/reward-machine/games/{game_name}.txt')
    with open(game_path, 'r') as f:
        game = f.read()
    return game


TEST_GAME_LIBRARY = {
    'on-chair-bug': load_game("building_on_bug"),
    'test-building': load_game("building_base"),
    'test-building-on-chair': load_game("building_on_chair"),
    'test-building-in-bin': load_game("building_in_bin"),
    'test-building-touches-wall': load_game("building_touches_wall"),
    'test-throwing': load_game("throw_to_bin"),
    'test-throw-to-wall': load_game("throw_at_wall"),
    'test-measure': load_game("throw_measure_dist"),
    'test-wall-bounce': load_game("throw_bounce_wall_bin"),
    'test-setup': load_game("throw_with_setup"),
    'test-external-scoring': load_game("throw_external_maximize"),
    'test-count-unique-positions': load_game("throw_to_bin_unique_positions"),
    'test-count-overlapping': load_game("building_count_overlapping"),
    'test-clean-room': load_game("game-27"),
    'test-agent-adjacent': load_game("test_agent_door_adjacent"),
    'throw-block-cache-test': load_game("throw_with_stacked_blocks"),
    'test-ball-from-bed': load_game("ball_to_bin_from_bed"),
    'test-block-stacking': load_game("block_stacking"),
}

TEST_CASES = [
    ('on-chair-bug', BLOCK_STACKING_TRACE, 0.0, {},),
    ('test-building', BLOCK_STACKING_TRACE, 10.0, {
        'blockInBuildingAtEnd': [
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CubeBlock|-02.96|+01.26|-01.72'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CubeBlock|-02.97|+01.26|-01.94'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CylinderBlock|-02.95|+01.62|-01.95'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CylinderBlock|-03.02|+01.62|-01.73'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'CubeBlock|-02.99|+01.26|-01.49'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'CylinderBlock|-02.97|+01.62|-01.50'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'PyramidBlock|-02.95|+01.61|-02.66'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'TallRectBlock|-02.95|+02.05|-02.31'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'TallRectBlock|-02.95|+02.05|-02.52'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'TallRectBlock|-02.95|+02.05|-02.72'}, start=2140, end=2140, measures={})
        ]
    },),
    ('test-building-on-chair', BLOCK_STACKING_TRACE, 12.0, {
        'blockInBuildingOnChairAtEnd': [
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'CubeBlock|-02.99|+01.26|-01.49', '?c': 'Chair|+02.73|00.00|-01.21'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'CylinderBlock|-02.97|+01.62|-01.50', '?c': 'Chair|+02.73|00.00|-01.21'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'PyramidBlock|-02.95|+01.61|-02.66', '?c': 'Chair|+02.73|00.00|-01.21'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'TallRectBlock|-02.95|+02.05|-02.31', '?c': 'Chair|+02.73|00.00|-01.21'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'TallRectBlock|-02.95|+02.05|-02.52', '?c': 'Chair|+02.73|00.00|-01.21'}, start=2140, end=2140, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'TallRectBlock|-02.95|+02.05|-02.72', '?c': 'Chair|+02.73|00.00|-01.21'}, start=2140, end=2140, measures={})
        ]
    },),
    ('test-building-in-bin', BUILDING_IN_TOUCH_TEST_TRACE, 4.0, {
        'blockInBuildingInBinAtEnd': [
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CubeBlock|+00.50|+01.61|-02.91', '?h': 'GarbageCan|-02.79|-00.03|-02.67'}, start=2039, end=2039, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CubeBlock|+00.70|+01.61|-02.91', '?h': 'GarbageCan|-02.79|-00.03|-02.67'}, start=2039, end=2039, measures={})
        ]
    },),
    ('test-building-touches-wall', BUILDING_IN_TOUCH_TEST_TRACE, 8.0, {
        'blockInBuildingTouchingWallAtEnd': [
            PreferenceSatisfaction(mapping={'?b': 'building_2', '?l': 'BridgeBlock|+00.63|+01.10|-02.91', '?w': 'north_wall'}, start=2039, end=2039, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_2', '?l': 'BridgeBlock|+01.03|+01.11|-02.88', '?w': 'north_wall'}, start=2039, end=2039, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_2', '?l': 'CylinderBlock|+00.93|+01.61|-02.89', '?w': 'north_wall'}, start=2039, end=2039, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_2', '?l': 'FlatRectBlock|+00.23|+01.66|-02.88', '?w': 'north_wall'}, start=2039, end=2039, measures={})
        ]
    },),
    ('test-throwing', BALL_TO_WALL_TO_BIN_TRACE, -1.2, {
        'throwBallToBin': [
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1958, end=2015, measures={}),
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2040, end=2151, measures={}),
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2374, end=2410, measures={})
        ],
        'throwAttempt': [
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=343, end=456, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=457, end=590, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=769, end=880, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=881, end=947, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=948, end=1019, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1020, end=1120, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1121, end=1209, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1210, end=1279, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1280, end=1370, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1371, end=1439, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1440, end=1502, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1503, end=1555, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1556, end=1656, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1699, end=1782, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1783, end=1868, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1869, end=1957, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1958, end=2015, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2040, end=2151, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2198, end=2293, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2294, end=2373, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2374, end=2410, measures={})
        ]
    },),
    ('test-throw-to-wall', BALL_TO_WALL_TO_BIN_TRACE, 23.0, {
        'throwToWall': [
            PreferenceSatisfaction(mapping={'?w': 'east_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=343, end=456, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=343, end=456, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'south_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=343, end=456, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=457, end=590, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=769, end=880, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=881, end=947, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=948, end=1019, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1020, end=1120, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1121, end=1209, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1210, end=1279, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1280, end=1370, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1371, end=1439, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1440, end=1502, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1503, end=1555, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1556, end=1656, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1699, end=1782, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1783, end=1868, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1869, end=1957, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1958, end=2015, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2040, end=2151, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2198, end=2293, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2294, end=2373, measures={}),
            PreferenceSatisfaction(mapping={'?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2374, end=2410, measures={})
        ],
    },),
    ('test-wall-bounce', THREE_WALL_TO_BIN_BOUNCES_TRACE, 30, {
        'throwToWallToBin' : [
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=1958, end=2015, measures={}),
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2040, end=2151, measures={}),
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?w': 'north_wall', '?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=2374, end=2410, measures={})
        ],
    },),
    ('test-measure', THREE_WALL_TO_BIN_BOUNCES_TRACE, 3.7705330066455316, {
        'throwToBinFromDistance' : [
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80', 'agent': 'agent', '?h': 'GarbageCan|+00.75|-00.03|-02.74'}, start=1958, end=2015, measures={'distance': 1.1817426840899055}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80', 'agent': 'agent', '?h': 'GarbageCan|+00.75|-00.03|-02.74'}, start=2040, end=2151, measures={'distance': 1.3097693013954452}),
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|+00.70|+01.11|-02.80', 'agent': 'agent', '?h': 'GarbageCan|+00.75|-00.03|-02.74'}, start=2374, end=2410, measures={'distance': 1.2790210211601807})
        ],
    },),
    ('test-setup', SETUP_TRACE, 2.0, {
        'throwToBin' : [
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.19|+01.13|-02.80'}, start=2542, end=2642, measures={}),
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.19|+01.13|-02.80'}, start=3572, end=3620, measures={})
        ],
    },),
    ('test-external-scoring', THROW_ALL_DODGEBALLS_TRACE, 4.0, {
        'throwAttempt' : [
            PreferenceSatisfaction(mapping={'?b': 'Dodgeball|+00.19|+01.13|-02.80'}, start=38, end=99, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'Dodgeball|+00.44|+01.13|-02.80'}, start=104, end=151, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=167, end=248, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=394, end=468, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'Dodgeball|+00.19|+01.13|-02.80'}, start=296, end=521, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=523, end=662, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'Dodgeball|+00.70|+01.11|-02.80'}, start=663, end=888, measures={})
        ],
    },),
    ('test-count-unique-positions', THROW_BALL_UNIQUE_POSITIONS_TRACE, 3.0, {
        'throwBallToBin' : [
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.19|+01.13|-02.80'}, start=963, end=1047, measures={}),
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.19|+01.13|-02.80'}, start=1220, end=1304, measures={}),
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.19|+01.13|-02.80'}, start=1483, end=1545, measures={}),
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.19|+01.13|-02.80'}, start=1647, end=1719, measures={})
        ],
    },),
    ('test-count-overlapping', COMPLEX_STACKING_TRACE, 3.0, {
        'blockPlacedInBuilding' : [
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CubeBlock|-02.97|+01.26|-01.94'}, start=31, end=391, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_4', '?l': 'CubeBlock|-02.97|+01.26|-01.94'}, start=391, end=638, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'CubeBlock|-02.97|+01.26|-01.94'}, start=638, end=706, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CubeBlock|-02.96|+01.26|-01.72'}, start=442, end=754, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_0', '?l': 'CubeBlock|-02.99|+01.26|-01.49'}, start=549, end=754, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'CubeBlock|-02.96|+01.26|-01.72'}, start=442, end=754, measures={}),
            PreferenceSatisfaction(mapping={'?b': 'building_1', '?l': 'CubeBlock|-02.99|+01.26|-01.49'}, start=549, end=754, measures={})
        ],
        'blockPickedUp' : [
            PreferenceSatisfaction(mapping={'?l': 'CubeBlock|-02.96|+01.26|-01.72'}, start=59, end=116, measures={}),
            PreferenceSatisfaction(mapping={'?l': 'CubeBlock|-02.99|+01.26|-01.49'}, start=59, end=214, measures={}),
            PreferenceSatisfaction(mapping={'?l': 'CubeBlock|-02.97|+01.26|-01.94'}, start=54, end=404, measures={}),
            PreferenceSatisfaction(mapping={'?l': 'CubeBlock|-02.96|+01.26|-01.72'}, start=117, end=530, measures={}),
            PreferenceSatisfaction(mapping={'?l': 'CubeBlock|-02.99|+01.26|-01.49'}, start=215, end=611, measures={}),
            PreferenceSatisfaction(mapping={'?l': 'CubeBlock|-02.97|+01.26|-01.94'}, start=405, end=681, measures={}),
            PreferenceSatisfaction(mapping={'?l': 'CubeBlock|-02.97|+01.26|-01.94'}, start=682, end=753, measures={})
        ],
    },),
    ('test-clean-room', CLEANUP_TRACE, 69.0, {
        'dodgeballsInPlace' : [
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.95|-00.03|-02.68', '?d': 'Dodgeball|-02.95|+01.29|-02.61'}, start=2988, end=2988, measures={}),
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.95|-00.03|-02.68', '?d': 'Dodgeball|-02.97|+01.29|-02.28'}, start=2988, end=2988, measures={})
        ],
        'blocksInPlace' : [
            PreferenceSatisfaction(mapping={'?s': 'Shelf|-02.97|+01.16|-01.72', 'west_wall': 'west_wall', '?c': 'CubeBlock|+00.20|+00.29|-02.83'}, start=2988, end=2988, measures={}),
            PreferenceSatisfaction(mapping={'?s': 'Shelf|-02.97|+01.16|-01.72', 'west_wall': 'west_wall', '?c': 'CubeBlock|-00.02|+00.10|-02.83'}, start=2988, end=2988, measures={}),
            PreferenceSatisfaction(mapping={'?s': 'Shelf|-02.97|+01.16|-02.47', 'west_wall': 'west_wall', '?c': 'CubeBlock|-00.02|+00.28|-02.83'}, start=2988, end=2988, measures={}),
            PreferenceSatisfaction(mapping={'?s': 'Shelf|-02.97|+01.16|-02.47', 'west_wall': 'west_wall', '?c': 'CubeBlock|-00.23|+00.28|-02.83'}, start=2988, end=2988, measures={}),
            PreferenceSatisfaction(mapping={'?s': 'Shelf|-02.97|+01.53|-02.47', 'west_wall': 'west_wall', '?c': 'CubeBlock|-00.24|+00.10|-02.83'}, start=2988, end=2988, measures={}),
            PreferenceSatisfaction(mapping={'?s': 'Shelf|-02.97|+01.53|-01.72', 'west_wall': 'west_wall', '?c': 'CubeBlock|+00.20|+00.10|-02.83'}, start=2988, end=2988, measures={}),
        ],
        'laptopAndBookInPlace' : [
            PreferenceSatisfaction(mapping={'?s': 'Shelf|+00.62|+01.01|-02.82', '?o': 'Book|+02.83|+00.41|-00.01'}, start=2988, end=2988, measures={}),
            PreferenceSatisfaction(mapping={'?s': 'Shelf|+00.62|+01.51|-02.82', '?o': 'Laptop|+03.04|+00.79|-02.28'}, start=2988, end=2988, measures={})
        ],
        'smallItemsInPlace' : [
            PreferenceSatisfaction(mapping={'?d': 'Drawer|-01.52|+00.41|+00.35', '?o': 'CellPhone|+02.96|+00.79|-00.93'}, start=2988, end=2988, measures={}),
            PreferenceSatisfaction(mapping={'?d': 'Drawer|-01.52|+00.41|+00.35', '?o': 'KeyChain|-01.62|+00.60|+00.41'}, start=2988, end=2988, measures={})
        ],
        'itemsTurnedOff' : [
            PreferenceSatisfaction(mapping={'?o': 'LightSwitch|-00.14|+01.33|+00.60'}, start=2988, end=2988, measures={}),
            PreferenceSatisfaction(mapping={'?o': 'Desktop|+03.10|+00.79|-01.24'}, start=2988, end=2988, measures={}),
            PreferenceSatisfaction(mapping={'?o': 'Laptop|+03.04|+00.79|-02.28'}, start=2988, end=2988, measures={})
        ],
    },),
    ('test-agent-adjacent', TEST_AGENT_DOOR_ADJACENT_TRACE, 1.0, {
        'throwAdjacentToDoor' : [
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|-02.97|+01.29|-02.28', 'agent': 'agent', 'door': 'door', 'floor': 'Floor|+00.00|+00.00|+00.00', 'bed': 'Bed|-02.46|00.00|-00.57'}, start=136, end=197, measures={})
        ],
    },),
    ('throw-block-cache-test', TEST_BLOCK_CACHE_TEST, 3.0, {
        'throwWithStackedBlocksVerI': [
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.95|-00.03|-02.68', '?b': 'CubeBlock|+00.20|+00.29|-02.83', '?d': 'Dodgeball|-02.97|+01.29|-02.28'}, start=313, end=546, measures={})
        ],
        'throwWithStackedBlocksVerII': [
            PreferenceSatisfaction(mapping={'?d': 'Dodgeball|-02.97|+01.29|-02.28', '?h': 'GarbageCan|+00.95|-00.03|-02.68', '?b': 'CubeBlock|+00.20|+00.29|-02.83'}, start=313, end=545, measures={})
        ],
        'throwWithStackedBlocksVerIII': [
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.95|-00.03|-02.68', '?b': 'CubeBlock|+00.20|+00.29|-02.83', '?d': 'Dodgeball|-02.97|+01.29|-02.28'}, start=313, end=545, measures={})
        ]
    },),
    ('test-ball-from-bed', BALL_TO_BIN_FROM_BED_TRACE, 1.0, {
        'ballToBinFromBed': [
            PreferenceSatisfaction(mapping={'?h': 'GarbageCan|+00.75|-00.03|-02.74', '?b': 'Dodgeball|+00.19|+01.13|-02.80', 'bed': 'Bed|-02.46|00.00|-00.57', 'agent': 'agent'}, start=992, end=1142, measures={}),
        ],
    },),
]


@pytest.mark.parametrize("game_key, trace_path, expected_score, expected_satisfactions", TEST_CASES)
def test_single_game(game_key: str, trace_path: typing.Union[str, pathlib.Path],
    expected_score: float,
    expected_satisfactions: typing.Optional[typing.Dict[str, typing.List[PreferenceSatisfaction]]],
    debug: bool = False, debug_building_handler: bool = False, debug_preference_handlers: bool = False):

    game_def = TEST_GAME_LIBRARY[game_key]

    game_handler = GameHandler(game_def)
    score = None

    if isinstance(trace_path, pathlib.Path):
        trace_path = trace_path.resolve().as_posix()

    for state, is_final in _load_trace(trace_path):
        state = FullState.from_state_dict(state)
        score = game_handler.process(state, is_final, # debug=debug,
            debug_building_handler=debug_building_handler,
            debug_preference_handlers=debug_preference_handlers)
        if score is not None:
            break

    score = game_handler.score(game_handler.scoring)
    print(score, expected_score)
    assert np.allclose(score, expected_score)

    if expected_satisfactions is not None:
        for pref_name, pref_satisfactions in expected_satisfactions.items():
            assert pref_name in game_handler.preference_satisfactions

            for pref_satisfaction in pref_satisfactions:
                assert pref_satisfaction in game_handler.preference_satisfactions[pref_name]

if __name__ == '__main__':
    print(__file__)
    sys.exit(pytest.main([__file__]))
