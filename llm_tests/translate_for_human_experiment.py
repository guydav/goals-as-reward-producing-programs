from datetime import datetime
import json
import os
import sys

import logging
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

import tatsu
from tqdm import tqdm

sys.path.append(os.path.abspath('../src'))
from ast_utils import load_games_from_file
from game_describer import GameDescriber, TABLE_HTML_TEMPLATE, STYLE_HTML, TRANSLATIONS_PATH, TRANSLATIONS_PATH_SPLIT_BY_SECTION

FILES_TO_TRANSLATE = [
    # "../notebooks/games_for_human_experiment/human_games.pddl",
    "../notebooks/games_for_human_experiment/human_cell_archive_games.pddl",
    # "../notebooks/games_for_human_experiment/novel_archive_cell_games.pddl",
]

# Whether to split the final translation output into sections
SPLIT_BY_SECTION = True
DEFAULT_GRAMMAR_PATH = "../dsl/dsl.ebnf"

CURRENT_DATE = datetime.today().strftime("%Y-%m-%d").replace("-", "_")
OPENAI_MODEL = "gpt-4-1106-preview"
SPLIT_PROMPT_TO_SYSTEM = True

grammar = open(DEFAULT_GRAMMAR_PATH).read()
grammar_parser = tatsu.compile(grammar)
game_describer = GameDescriber(openai_model_str=OPENAI_MODEL, split_prompt_to_system=SPLIT_PROMPT_TO_SYSTEM)

for filepath in FILES_TO_TRANSLATE:
    name = filepath.split("/")[-1].split(".")[0]

    game_texts = []
    keys = []

    with open(filepath, "r") as f:
        for line in f.readlines():
            if line.startswith(";"):
                cur_game = ""
                game_texts.append(cur_game)

                key_str = line.lower().split("key ")[1][1:-2]
                keys.append(tuple([int(val) for val in key_str.split(", ")]))

            else:
                game_texts[-1] += line

    data = {}
    per_game_htmls = []
    for key, game in tqdm(zip(keys, game_texts), total=len(game_texts), desc=f"Translating {name}"):
        descriptions_by_stage = []

        stage_0_descriptions = game_describer.describe_stage_0(game, format_for_html=True)
        descriptions_by_stage.append(stage_0_descriptions)

        stage_1_descriptions = game_describer.describe_stage_1(game)
        descriptions_by_stage.append(stage_1_descriptions)

        translations_path = TRANSLATIONS_PATH_SPLIT_BY_SECTION if SPLIT_BY_SECTION else TRANSLATIONS_PATH

        stage_2_descriptions = game_describer.describe_stage_2(game, stage_1_descriptions, translations_path=translations_path)
        descriptions_by_stage.append(stage_2_descriptions)

        stage_3_description = game_describer.describe_stage_3(game, stage_2_descriptions, translations_path=translations_path)
        descriptions_by_stage.append((stage_3_description, "", "", ""))

        key = str(tuple(key))
        data[key] = stage_3_description

        table_html = game_describer._prepare_data_for_html_display(descriptions_by_stage, key=str(key))
        per_game_htmls.append(table_html)

        # Save as json
        with open(f"./translations/{name}_translations_{'split_' if SPLIT_BY_SECTION else ''}{CURRENT_DATE}.json", "w") as file:
            json.dump(data, file, indent=4)

        # Save as html
        full_html = TABLE_HTML_TEMPLATE.format(STYLE_HTML, "\n".join(per_game_htmls))
        output_filename = f"./translations/{name}_translations_{'split_' if SPLIT_BY_SECTION else ''}{CURRENT_DATE}.html"
        with open(output_filename, "w") as file:
            file.write(full_html)