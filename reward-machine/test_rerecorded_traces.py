import json
import os
import pathlib
import sys
from tqdm import tqdm

from game_handler import GameHandler
from utils import FullState, get_project_dir
from manual_run import _load_trace, load_game

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from ast_printer import ast_section_to_string

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
    'game-27': load_game("game-27"),
    'test-door': load_game("test_door"),
    'test-agent-door-adjacent': load_game("test_agent_door_adjacent"),
    'throw-block-cache-test': load_game("throw_with_stacked_blocks"),
}

TEST_TRACE_NAMES = ["throw_ball_to_bin_unique_positions", "setup_test_trace", "building_castle",
                    "throw_all_dodgeballs", "stack_3_cube_blocks", "three_wall_to_bin_bounces",
                    "complex_stacking_trace"]

TEST_GAME_NAMES = ["test-count-unique-positions", "test-setup", "test-building",
                    "test-external-scoring", "test-building", "test-wall-bounce",
                    "test-count-overlapping"]

for trace, game in zip(TEST_TRACE_NAMES, TEST_GAME_NAMES):
    if trace != "three_wall_to_bin_bounces":
        continue
    base_trace_path = pathlib.Path(f"{get_project_dir()}/reward-machine/traces/{trace}.json")
    rerecorded_trace_path = pathlib.Path(f"{get_project_dir()}/reward-machine/traces/{trace}-rerecorded.json")

    print(f"\n\n==========COMPARING {trace.upper()} TRACES ON GAME {game.upper()}==========")
    for i, path in enumerate([base_trace_path, rerecorded_trace_path]):
        game_handler = GameHandler(TEST_GAME_LIBRARY[game])
        score = None

        trace = list(_load_trace(path))

        for idx, (state, is_final) in tqdm(enumerate(trace), total=len(trace), desc=f"Processing trace", leave=False):
            state = FullState.from_state_dict(state)

            score = game_handler.process(state, is_final, debug_preference_handlers=False, debug_building_handler=False)
            if score is not None:
                break

        score = game_handler.score(game_handler.scoring)


        print(f"\n-----Satisfactions for {'RE-RECORDED' if i == 1 else 'ORIGINAL'} trace-----")
        used_mappings = set()
        for key, val in game_handler.preference_satisfactions.items():
            print(key + ":")
            for sat in val:
                print(f"\t{sat}")


        # handler = game_handler.preference_handlers["throwBallToBin"]
        # for partial in handler.partial_preference_satisfactions:
        #     print("\nPartial satisfaction:")
        #     print(f"\tMapping: {partial.mapping}")
        #     print(f"\tCurrent predicate: {ast_section_to_string(partial.current_predicate, '(:constraints') if partial.current_predicate else None}")
        #     print(f"\tNext predicate: {ast_section_to_string(partial.next_predicate, '(:constraints') if partial.next_predicate else None}")
        #     print(f"\tStart: {partial.start}")

            

    # break