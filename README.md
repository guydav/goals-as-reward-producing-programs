# Goals as Reward-Producing Programs
A cleaned up version of the long-running development repository for our paper [Goals as Reward-Producing Programs](https://exps.gureckislab.org/guydav/goal_programs_viewer/main/).

Quick links:
* [Project webpage](https://exps.gureckislab.org/guydav/goal_programs_viewer/main/)
* [Paper](https://arxiv.org/abs/2405.13242)
* [Raw participant responses in our behavioral experiment](https://github.com/guydav/goals-as-reward-producing-programs/blob/main/data/interactive_beta.csv)
* [Participant-created games as programs in our DSL](https://github.com/guydav/goals-as-reward-producing-programs/blob/main/dsl/interactive-beta.pddl)
* [Our DSL in an eBNF](https://github.com/guydav/goals-as-reward-producing-programs/blob/main/dsl/dsl.ebnf)
* [Code for our game creation behavioral experiment](https://github.com/guydav/game-creation-behavioral-experiment)
* [Code for our game human evaluation experiment](https://github.com/guydav/game-fitness-judgements)

## Reproduction
To reproduce analyses reported in the paper, see the notebooks in the [`reproduction_notebooks`](https://github.com/guydav/goals-as-reward-producing-programs/tree/main/reproduction_notebooks) folder.
We provide three notebooks:
* [`behavioral_analyses.ipynb`](https://github.com/guydav/goals-as-reward-producing-programs/blob/main/reproduction_notebooks/behavioral_analyses.ipynb) reproduces anaylses of our behavioral data, reported in Figure 2.
* [`human_evaluation_summary.ipynb`](https://github.com/guydav/goals-as-reward-producing-programs/blob/main/reproduction_notebooks/human_evaluations_summary.ipynb) reproduces summary analyses of our human evaluations, reported in Tables 1 and SI-2.
* [`human_evaluations_mixed_models.ipynb`](https://github.com/guydav/goals-as-reward-producing-programs/blob/main/reproduction_notebooks/human_evaluations_mixed_models.ipynb) reproduces the mixed effect model analyses of our human evaluations, reported in Tables ED-1, SI-3, and SI-4.

## Viewing games
The easiest way to view games including in the human evaluation, including their back-translations to natural language, is to use the [project webpage](https://exps.gureckislab.org/guydav/goal_programs_viewer/main).

Otherwise, you can also use the [`game_viewer.ipynb`](https://github.com/guydav/goals-as-reward-producing-programs/blob/main/reproduction_notebooks/game_viewer.ipynb) to view any game from our participant-created dataset or model productions.

## Setup
Create a conda environment, e.g. `conda env create -n game-gen python=3.10` (we worked with Python 3.10 on OS X and Ubuntu, though other versions should work, too).

Activate the environment with `conda activate game-gen`.

Install the requirements with `pip install -r requirements.txt`.

Running the R notebook `human_evaluations_mixed_models.ipynb` requires installing the R kernel for Jupyter, we use VS Code's integration, which requires following a few [additional setup steps](https://saturncloud.io/blog/how-to-use-jupyter-r-kernel-with-visual-studio-code/).

All in all, the setup should not take particularly long if you already have VS Code and conda installed.

## Running the model
Running the model includes several steps, depending on what exactly you choose to modify from our current model. As a result, we provide a list of steps in a separate file, [`MODEL_README.md`](https://github.com/guydav/goals-as-reward-producing-programs/blob/main/MODEL_README.md).


