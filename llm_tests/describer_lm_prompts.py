import csv

###
# Prompts used for getting GPT-X to generate natural language (ish) descriptions of games. Prompts are broken down according to which 'stage' they map from:
# -Stage 0: raw game code
# -Stage 1: templated game description
# -Stage 2: natural language conversion of template
# -Stage 3: natural language description of game
###

SECTION_PROMPT_SUFFIX = """

Now, convert the following description:
### INITIAL DESCRIPTION:
{0}

### CONVERTED DESCRIPTION:"""


SHARED_PROMPT = "but you may rewrite and reorder the information in any way you think is necessary in order for a human to understand it. Use simple language and verbs that would be familiar to a human who has never played this game before."


SETUP_PROMPT = f"""Your task is to convert a templated description of a game's setup into a natural language description. Do not change the content of the template, {SHARED_PROMPT}
Use the following examples as a guide:
{{0}}"""

PREFERENCES_PROMPT = f"""Your task is to convert a templated description of a game's rules (expressed as "preferences") into a natural language description. Do not change the content of the template, {SHARED_PROMPT}
Use the following examples as a guide:
{{0}}"""

TERMINAL_PROMPT = f"""Your task is to convert a templated description of a game's terminal conditions into a natural language description. Do not change the content of the template, {SHARED_PROMPT}
Use the following examples as a guide:
{{0}}"""

SCORING_PROMPT= f"""Your task is to convert a templated description of a game's scoring conditions into a natural language description. Do not change the content of the template, {SHARED_PROMPT}
Use the following examples as a guide:
{{0}}"""

COMPLETE_GAME_PROMPT = f"""Your task is to combine and simplify the description of a game's rules. Do not change the content of the rules by either adding or removing information, {SHARED_PROMPT} DO describe preferences carefully, such that a player reading the description can easily play the game. DO NOT include explicit references to a game's preferences (i.e. "Preference 1" or "Preference 2"). DO NOT include descriptions of setup or terminal conditions if they do not appear in the game.
Use the following examples as a guide:
{{0}}"""

COMPLETE_GAME_VALIDATION_PROMPT = """Please edit the following game description to remove all explicit references to a game's preferences (i.e. "Preference 1" or "Preference 2") and replace them with their corresponding natural language descriptions. DO NOT change any other content of the game description.
Here is the game's information:
{0}

And here is the game description to re-write:
{1}
"""


def compile_prompts_from_data(initial_stage: int,
                              final_stage: int,
                              translations_path: str):

    # Load in the data
    with open(translations_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)[1:]

    setup_content = ""
    preferences_content = ""
    terminal_content = ""
    scoring_content = ""

    complete_content = ""
    game_overall_prefix = ""
    game_overall_suffix = ""

    for idx, row in enumerate(data):

        stages = row[1:]

        # Identifier
        if idx%5 == 0:

            # Add overall description to complete description and reset the overall description
            complete_content += game_overall_prefix + game_overall_suffix

            game_overall_prefix = "\n\n### INITIAL DESCRIPTION:"
            game_overall_suffix = "\n\n### CONVERTED DESCRIPTION\n"
            continue

        # Setup (optional)
        elif idx%5 == 1:
            if stages[initial_stage] != "":
                setup_content += f"### INITIAL DESCRIPTION:\n{stages[initial_stage]}\n\n### CONVERTED DESCRIPTION:\n{stages[final_stage]}\n\n"

        # Preferences (required)
        elif idx%5 == 2:
            preferences_content += f"### INITIAL DESCRIPTION:\n{stages[initial_stage]}\n\n### CONVERTED DESCRIPTION:\n{stages[final_stage]}\n\n"

        # Terminal (optional)
        elif idx%5 == 3:
            if stages[initial_stage] != "":
                terminal_content += f"### INITIAL DESCRIPTION:\n{stages[initial_stage]}\n\n### CONVERTED DESCRIPTION:\n{stages[final_stage]}\n\n"

        # Scoring (required)
        elif idx%5 == 4:
            scoring_content += f"### INITIAL DESCRIPTION:\n{stages[initial_stage]}\n\n### CONVERTED DESCRIPTION:\n{stages[final_stage]}\n\n"

        if stages[initial_stage] != "":
            game_overall_prefix += f"\n\n{stages[initial_stage]}"
        game_overall_suffix += f"{stages[final_stage]}"

    # Compile the prompts
    setup_prompt = SETUP_PROMPT.format(setup_content) + SECTION_PROMPT_SUFFIX
    preferences_prompt = PREFERENCES_PROMPT.format(preferences_content) + SECTION_PROMPT_SUFFIX
    terminal_prompt = TERMINAL_PROMPT.format(terminal_content) + SECTION_PROMPT_SUFFIX
    scoring_prompt = SCORING_PROMPT.format(scoring_content) + SECTION_PROMPT_SUFFIX
    complete_game_prompt = COMPLETE_GAME_PROMPT.format(complete_content) + SECTION_PROMPT_SUFFIX

    return setup_prompt, preferences_prompt, terminal_prompt, scoring_prompt, complete_game_prompt

if __name__ == '__main__':
    a, b, c, d, e = compile_prompts_from_data(2, 3, "./llm_tests/selected_human_and_map_elites_translations_split_by_section.csv")
    breakpoint()
