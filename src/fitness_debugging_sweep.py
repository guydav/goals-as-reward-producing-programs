import itertools
import gzip
import os
import pickle

import numpy as np
import pandas as pd
import torch
import tqdm

import fitness_energy_utils as utils
from fitness_energy_utils import NON_FEATURE_COLUMNS


fitness_df = utils.load_fitness_data('../data/fitness_features_1024_regrowths.csv.gz')
print(fitness_df.src_file.unique())
print(fitness_df.shape)
original_game_counts = fitness_df.groupby('original_game_name').src_file.count().value_counts()
if len(original_game_counts) == 1:
    print(f'All original games have {original_game_counts.index[0] - 1} regrowths')  # type: ignore
else:
    print('Some original games have different numbers of regrowths: {original_game_counts}')
fitness_df.head()


def get_features_by_abs_diff_threshold(diffs: pd.Series, score_threshold: float):
    feature_columns = list(diffs[diffs >= score_threshold].index)

    remove_all_ngram_scores = []
    for score_type in ('full', 'setup', 'constraints', 'terminal', 'scoring'):
        col_names = sorted([c for c in feature_columns if c.startswith(f'ast_ngram_{score_type}') and c.endswith('_score')])

        if score_type not in remove_all_ngram_scores:
            col_names = col_names[:-1]

        for col in col_names:
            feature_columns.remove(col)

    return feature_columns


BETA = 1.0
N_WORKERS = 4
CHUNKSIZE = 10
RANDOM_SEED = 55

scaler_kwargs = dict(passthrough=True)
model_kwargs = dict()
train_kwargs = dict(
    loss_function=utils.fitness_softmin_loss,
    k=1024,
    lr=4e-3,
    beta=BETA,
    negative_score_reduction='none',
    # n_epochs=10000,
    shuffle_negatives=True,
    bias_init_margin_ratio=0.01,
    device=torch.device('cuda:0'),
    # regularizer=regularizer,
    split_validation_from_train=True,
    )

sweep_param_grid = dict(
    n_epochs=[5000, 10000, 15000],
    # patience_epochs=[10000],  # range(50, 300, 50),
    use_lr_scheduler=[False, True],
    batch_size=[1, 2, 4, 8, 16], # batch_size=[1, 2, 4, 8, 16],
    score_threshold=[0, 0.005, 0.01, 0.02, 0.03, 0.04], # score_threshold=[0, 0.005, 0.01, 0.02, 0.03, 0.04],
)

scoring = utils.build_multiple_scoring_function(
    [utils.wrap_loss_function_to_metric(utils.fitness_sofmin_loss_positive_negative_split, dict(beta=BETA), True),  # type: ignore
     utils.evaluate_fitness_overall_ecdf, utils.evaluate_fitness_single_game_rank, utils.evaluate_fitness_single_game_min_rank,
     utils.wrap_loss_function_to_metric(utils.energy_of_negative_at_quantile, dict(quantile=0.01), True),  # type: ignore
     utils.wrap_loss_function_to_metric(utils.energy_of_negative_at_quantile, dict(quantile=0.05), True),  # type: ignore
     ],
    ['loss', 'overall_ecdf', 'single_game_rank', 'single_game_min_rank', 'energy_of_negative@1%', 'energy_of_negative@5%'],
)

mean_features_by_real = fitness_df[['real'] + [c for c in fitness_df.columns if c not in NON_FEATURE_COLUMNS]].groupby('real').mean()
feature_diffs = mean_features_by_real.loc[1] - mean_features_by_real.loc[0]
abs_diffs = feature_diffs.abs()  # .sort_values(ascending=False)

sweep_models = {}
sweep_results = {}
sweep_losses = {}

def fit_configuration(setting_kwargs):
    setting_key = list(setting_kwargs.values())

    setting_train_kwargs = train_kwargs.copy()
    score_threshold = setting_kwargs.pop('score_threshold')
    setting_train_kwargs.update(setting_kwargs)

    if 'patience_epochs' not in setting_train_kwargs:
        setting_train_kwargs['patience_epochs'] = setting_train_kwargs['n_epochs']

    feature_columns = get_features_by_abs_diff_threshold(abs_diffs, score_threshold)

    model, _, results = utils.initialize_and_fit_model(
        fitness_df, split_test_set=True, feature_columns=feature_columns,
        random_seed=RANDOM_SEED,
        scaler_kwargs=scaler_kwargs, model_kwargs=model_kwargs, train_kwargs=setting_train_kwargs,
        scoring_function=scoring,
    )

    setting_key.append(len(feature_columns))
    setting_key = tuple(setting_key)
    model.named_steps['fitness'].model.to(torch.device('cpu'))
    sweep_models[setting_key] = model
    sweep_results[setting_key] = results
    sweep_losses[setting_key] = model.named_steps['fitness'].losses


def param_combination_iterator():
    for combination in itertools.product(*sweep_param_grid.values()):  # type: ignore
        yield dict(zip(sweep_param_grid.keys(), combination))


for setting_kwargs in tqdm.tqdm(param_combination_iterator(), total=np.product([len(v) for v in sweep_param_grid.values()])):  # type: ignore
    fit_configuration(setting_kwargs)

sweep_results_df = []

try:
    KEY_HEADERS = list(sweep_param_grid.keys()) + ['n_features']
    example_values = next(iter(sweep_results.values()))
    VALUE_HEADERS = [f'{outer_key}_{inner_key}' for outer_key in example_values for inner_key in example_values[outer_key]]

    rows = [list(key) + [results[outer_key][inner_key] for outer_key in results for inner_key in results[outer_key]]
            for key, results in sweep_results.items()]

    sweep_results_df = pd.DataFrame(rows, columns=KEY_HEADERS + VALUE_HEADERS)
    sweep_results_df = sweep_results_df.assign(**{c: sweep_results_df[c].abs() for c in sweep_results_df.columns if 'ecdf' in c or 'loss' in c or 'energy_of_negative' in c},
                                            use_lr_scheduler=sweep_results_df.use_lr_scheduler.astype(int))
except:
    pass
# sweep_results_df.head()

RUN_NAME = 'n_epochs'

os.makedirs('../tmp', exist_ok=True)
try:
    with gzip.open(f'../tmp/sweep_results_large_{RANDOM_SEED}_{RUN_NAME}.pkl.gz', 'wb') as f:
        pickle.dump(dict(sweep_results_df=sweep_results_df, sweep_models=sweep_models, sweep_losses=sweep_losses, sweep_results=sweep_results), f, protocol=pickle.HIGHEST_PROTOCOL)

except:
    with gzip.open(f'../tmp/sweep_results_large_no_models_{RANDOM_SEED}_{RUN_NAME}.pkl.gz', 'wb') as f:
        pickle.dump(dict(sweep_results_df=sweep_results_df, sweep_losses=sweep_losses, sweep_results=sweep_results), f, protocol=pickle.HIGHEST_PROTOCOL)
