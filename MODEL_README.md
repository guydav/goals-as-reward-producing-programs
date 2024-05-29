# Model Running Readme 
Depending on if (and which) steps you choose to modify, running the model includes several steps. We enumerate them below:

## 1. Negative corruption sampling
Assuming you have your real examples saved in `./dsl/my-games.pddl`, you can run the follow script to generate regrowths, and then cache their ASTs (instead of having to parse them from strings each time) for further use:

```bash
#!/bin/bash

OUTPUTH_PATH="./dsl/ast-real-regrowth-samples-1024.pddl.gz"

python src/ast_counter_sampler.py \
    --test-files "./dsl/my-games.pddl" \
    --sampling-method "regrowth" \
    --parse-counter \
    -n 1024 \
    --sample-parallel \
    --parallel-n-workers 15 \
    --parallel-chunksize 32 \
    --sample-tqdm \
    --save-samples \
    --prior-count 2 \
    --prior-count 12 \
    --prior-count 22 \
    --max-sample-depth 16 \
    --max-sample-nodes 128 \
    --fix-contexts \
    --samples-output-path $OUTPUTH_PATH

sleep 10

python src/parse_dsl.py \
    -t $OUTPUTH_PATH \
    --expected-total 100352 \
    --force-rebuild-cache
```

## 1.5 Re-run the n-gram model
If your distribution of games is drastically different than ours, you will probably want to fit the n-gram model to your data. To do so, run the following script:

```bash
#!/bin/bash

python src/fitness_ngram_models.py \
    --test-files "./dsl/my-games.pddl" \
    --from-asts \
    -n 5 
```

If you do, update `latest_model_paths.py`, setting `LATEST_NGRAM_MODEL_PATH` to the resultant file.

## 2. Fitness featurizer creation and feature extraction
Next, you will want to create a fitness featurizer to these samples and use it to extract feature. To do so, run the following script:

```bash
#!/bin/bash

python src/fitness_features.py \
    --test-files "./dsl/my-games.pddl" \
    --test-files "./dsl/ast-real-regrowth-samples-1024.pddl.gz" \
    --n-workers 15  \
    --chunksize 670 

```

After this, update `latest_model_paths.py`, setting `LATEST_REAL_GAMES_PATH`, `LATEST_FITNESS_FEATURIZER_PATH`, and `LATEST_FITNESS_FEATURES` (if you changed it from its default value). 

## 3. Fit the fitness function
Next, you will want to fit the fitness function to the features extracted in the previous step. To do so, run the following script:

```bash
#!/bin/bash

python run/fitness_models_cv.py \
    --output-name "fitness_sweep_in_data_prop_L2" \
    --random-seed 42 \
    --cv-settings-json "run/fitness_cv_settings.json" \
    --train-kwargs-json-key "train_kwargs_l2" \
    --param-grid-json-key "param_grid_l2" \
    --cv-kwargs-json-key "cv_kwargs_refit_loss"
```

After this, update `latest_model_paths.py`, setting `LATEST_FITNESS_FUNCTION_DATE_ID` to the resultant file.

## 4. Run the model
Finally, you can run the model. To do so, run the following script:

```bash
#!/bin/bash

BEHAVIORAL_FEATURES_KEY="exemplar_preferences_bc_max_prefs_expected_values"
RANDOM_SEED=42
OUTPUT_NAME="map_elites_minimal_counting_grammar_use_forall_L2" 
N_ITER=8192


python src/evolutionary_sampler.py \
    --max-sample-depth 16 \
    --max-sample-nodes 128 \
    --sampler-filter-func-key "no_identical_preferences" \
    --sample-parallel \
    --parallel-n-workers 15 \
    --parallel-chunksize 50 \
    --parallel-maxtasksperchild 6400 \
    --should-tqdm \
    --initial-proposal-type 0 \
    --population-size 128 \
    --map-elites-initial-candidate-pool-size 1024 \
    --map-elites-generation-size 750 \
    --n-steps $N_ITER \
    --sampler-type "map_elites" \
    --map-elites-initialization-strategy 3 \
    --map-elites-weight-strategy 0 \
    --map-elites-custom-behavioral-features-key $BEHAVIORAL_FEATURES_KEY \
    --map-elites-use-cognitive-operators \
    --map-elites-use-crossover \
    --output-name $OUTPUT_NAME \
    --sampler-prior-count 2 \
    --sampler-prior-count 12 \
    --sampler-prior-count 22 \
    --save-interval 256 \
    --verbose 1 \
    --random-seed $RANDOM_SEED \
    --wandb \
    --resume \
    --resume-max-days-back 4
    
```

## 4.5 (Optional) run the reward machine trace filter
If you then want to compute the reward machine trace filter results for each game, you can run the following script:

```bash

BEHAVIORAL_FEATURES_KEY="exemplar_preferences_bc_max_prefs_expected_values"
RANDOM_SEED=42
OUTPUT_NAME="map_elites_minimal_counting_grammar_use_forall_L2" 
N_ITER=8192
TODAY=`date +'%Y_%m_%d'`
MAP_ELITES_FULL_OUTPUT_PATH="${OUTPUT_NAME}_${BEHAVIORAL_FEATURES_KEY}_uniform_seed_${RANDOM_SEED}_gen_${N_ITER}_final_${TODAY}"
echo $MAP_ELITES_FULL_OUTPUT_PATH

python reward-machine/reward_machine_trace_filter.py \
   --map-elites-model-name $MAP_ELITES_FULL_OUTPUT_PATH \
   --tqdm \
   --max-traces-per-game 400 \
   --n-workers 15 \
   --base-trace-path "/misc/vlgscratch4/LakeGroup/guy/participant-traces" \
   --chunksize 1 \
   --save-interval 1 \
   --use-only-database-nonconfirmed-traces \
   --dont-sort-keys-by-traces \
   --copy-traces-to-tmp
```
