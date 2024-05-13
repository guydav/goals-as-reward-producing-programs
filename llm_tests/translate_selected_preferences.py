from datetime import datetime
import json
import os
import pickle
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

DEFAULT_GRAMMAR_PATH = "../dsl/dsl.ebnf"
OPENAI_MODEL = "gpt-4-1106-preview"
SPLIT_PROMPT_TO_SYSTEM = True

grammar = open(DEFAULT_GRAMMAR_PATH).read()
grammar_parser = tatsu.compile(grammar)
game_describer = GameDescriber(openai_model_str=OPENAI_MODEL, split_prompt_to_system=SPLIT_PROMPT_TO_SYSTEM)

with open('../notebooks/games_for_human_experiment/exemplar_preferences.pkl', 'rb') as f:
    exemplar_preference_dict = pickle.load(f)

results = {}
for key, ast in exemplar_preference_dict.items():
    stage_1 = game_describer._describe_preference(ast)
    _, stage_2, _, _ = game_describer.describe_stage_2(ast, stage_1_descriptions=("", stage_1, "", ""))
    
    output = {
        "stage_1": stage_1,
        "stage_2": stage_2,
    }

    results[key] = output

with open("./translations/selected_preferences.json", "w") as f:
    json.dump(results, f, indent=4)