import json
import pathlib
import typing

from game_handler import GameHandler
from utils import FullState, get_project_dir


BLOCK_STACKING_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/block_stacking_test_trace.json')
THROWING_BALLS_AT_WALL_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/throwing_balls_at_wall.json')
SIMPLE_STACKING_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/simple_stacking_trace.json')
THREE_WALL_TO_BIN_BOUNCES_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/three_wall_to_bin_bounces.json')
SETUP_TEST_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/setup_test_trace.json')
CASTLE_TEST_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/building_castle.json')
BUILDING_IN_TOUCH_TEST_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/weZ1UVzKNaiTjaqu0DGI-preCreateGame-buildings-in-touching.json')
THROW_ALL_DODGEBALLS_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/throw_all_dodgeballs.json')
THROW_BALL_UNIQUE_POSITIONS_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/throw_ball_to_bin_unique_positions.json')
STACK_THREE_CUBES_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/stack_3_cube_blocks.json')
COMPLEX_STACKING_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/complex_stacking_trace.json')
GAME_6_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/game-6.json')
GAME_15_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/game-15.json')
GAME_27_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/game-27.json')
UPDATED_GAME_27_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/updated-game-27.json')
TEST_DOOR_AND_RUG_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/test_door_and_rug_collision.json')
TEST_AGENT_DOOR_ADJACENT_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/agent_door_adjacent.json')
THROW_WITH_STACKED_BLOCKS_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/throw_with_stacked_blocks.json')

# CLEANUP_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/qK8hfQE9E97kZMDdL4Hv-preCreateGame.json')
CLEANUP_TRACE = pathlib.Path(get_project_dir() + '/reward-machine/traces/qK8hfQE9E97kZMDdL4Hv-preCreateGame-rerecorded.json')
# TEST_BLOCK_CACHE_TEST = pathlib.Path(get_project_dir() + '/reward-machine/traces/otcaCEGfUhzEfGy72Qm8-preCreateGame.json')
TEST_BLOCK_CACHE_TEST = pathlib.Path(get_project_dir() + '/reward-machine/traces/otcaCEGfUhzEfGy72Qm8-preCreateGame-rerecorded.json')

REPLAY_NESTING_KEYS = (
    'participants-v2-develop',
    '17tSEDmCvGp1uKVEh5iq',
    'subCollection',
    'participants-v2-develop/17tSEDmCvGp1uKVEh5iq/replay-preCreateGame'
)

def _load_trace(path: str, replay_nesting_keys: typing.Optional[typing.Sequence[str]] = REPLAY_NESTING_KEYS):
    with open(path, 'r') as f:
        trace = json.load(f)
        if not isinstance(trace, list) and 'replay' in trace:
            trace = trace['replay']

    simple = isinstance(trace, list)

    if not simple and replay_nesting_keys is None:
        raise ValueError('Must provide replay_nesting_keys when not using simple mode')

    if simple:
        for idx, event in enumerate(trace):
            yield (event, idx == len(trace) - 1)

    else:
        replay_nesting_keys = typing.cast(typing.Sequence[str], replay_nesting_keys)
        for key in replay_nesting_keys:
            trace = trace[key]

        assert(all([key.startswith('batch-') for key in trace.keys()]))

        for batch_idx in range(len(trace)):
            batch = trace[f'batch-{batch_idx}']
            for idx, event in enumerate(batch['events']):
                yield (event, (idx == len(batch['events']) - 1) and (batch_idx == len(trace) - 1)) # make sure we're in the last batch and the last event

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
    'test-score-once-per-external': load_game("throw_count_once_per_external_objects"),
    'test-count-unique-positions': load_game("throw_to_bin_unique_positions"),
    'test-count-same-positions': load_game("throw_to_bin_same_positions"),
    'test-count-overlapping': load_game("building_count_overlapping"),
    'test-adjacent': load_game("adjacent_at_end"),
    'test-ball-from-bed': load_game("ball_to_bin_from_bed"),
    'game-6': load_game("game-6"),
    'game-7': load_game("game-7"),
    'game-15': load_game("game-15"),
    'test-clean-room': load_game("game-27"),
    'test-door': load_game("test_door"),
    'test-agent-door-adjacent': load_game("test_agent_door_adjacent"),
    'throw-block-cache-test': load_game("throw_with_stacked_blocks"),
}

if __name__ == "__main__":


    # game = TEST_GAME_LIBRARY['throw-block-cache-test']
    game = TEST_GAME_LIBRARY['test-clean-room']
    game_handler = GameHandler(game)
    # trace_path = TEST_BLOCK_CACHE_TEST.resolve().as_posix()
    trace_path = CLEANUP_TRACE.resolve().as_posix()
    # trace_path = BUILDING_IN_TOUCH_TEST_TRACE.resolve().as_posix()
    # trace_path = '/Users/guydavidson/Downloads/hRlEvjz5alx99uwENncs-preCreateGame.json'

    score = None

    for idx, (state, is_final) in enumerate(_load_trace(trace_path)):
        # print(f"\n\n================================PROCESSING STATE {idx} ================================")
        state = FullState.from_state_dict(state)
        score = game_handler.process(state, is_final, debug_preference_handlers=False, debug_building_handler=False)
        if score is not None:
            break

    score = game_handler.score(game_handler.scoring)

    if score is not None:
        print("\n\nSCORE ACHIEVED:", score)

    print("\nPREFERENCE SATISFACTIONS")
    used_mappings = set()
    for key, val in game_handler.preference_satisfactions.items():
        print(key + ":")
        for sat in val:
            print(f"\t{sat}")
