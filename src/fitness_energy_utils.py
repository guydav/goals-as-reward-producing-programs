from datetime import datetime, timedelta
from collections import defaultdict
import copy
from dataclasses import dataclass
from difflib import HtmlDiff
import gzip
from itertools import zip_longest, combinations
import logging
import os
import pickle
import typing
import sys

from IPython.display import display, Markdown, HTML  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from statsmodels.distributions.empirical_distribution import ECDF
from tabulate import tabulate
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
from tqdm import tqdm

from fitness_features_preprocessing import NON_FEATURE_COLUMNS
import latest_model_paths


class LevelFilter(logging.Filter):
    def __init__(self, level, name: str = ""):
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging_handler_out = logging.StreamHandler(sys.stdout)
logging_handler_out.setLevel(logging.DEBUG)
logging_handler_out.addFilter(LevelFilter(logging.INFO))
logger.addHandler(logging_handler_out)

logging_handler_err = logging.StreamHandler(sys.stderr)
logging_handler_err.setLevel(logging.WARNING)
logger.addHandler(logging_handler_err)


def _find_nth(text, target, n):
    start = text.find(target)
    while start >= 0 and n > 1:
        start = text.find(target, start+len(target))
        n -= 1
    return start


def _add_original_game_name_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(original_game_name=df.game_name)  # real=fitness_df.src_file == 'interactive-beta.pddl',
    df.original_game_name.where(
        df.game_name.apply(lambda s: (s.count('-') <= 1) or (s.startswith('game-id') and s.count('-') >= 2)),
        df.original_game_name.apply(lambda s: s[:_find_nth(s, '-', 2)]),
        inplace=True)

    return df


def load_fitness_data(path: str = latest_model_paths.LATEST_FITNESS_FEATURES) -> pd.DataFrame:
    fitness_df = pd.read_csv(path)
    return process_fitness_df(fitness_df)


def process_fitness_df(fitness_df: pd.DataFrame) -> pd.DataFrame:
    fitness_df = _add_original_game_name_column(fitness_df)
    fitness_df.columns = [c.replace(' ', '_').replace('(:', '') for c in fitness_df.columns]
    fitness_df = fitness_df.assign(**{c: fitness_df[c].astype('int') for c in fitness_df.columns if fitness_df.dtypes[c] == bool})
    fitness_df = fitness_df[list(fitness_df.columns[:4]) + list(fitness_df.columns[-1:]) + list(fitness_df.columns[4:-1])]
    return fitness_df


MODELS_FOLDER = 'models'
DEFAULT_SAVE_MODEL_NAME = 'cv_fitness_model'
SAVE_MODEL_KEY = 'model'
BUG_SAVE_MODEL_KEY = 'moodel'
SAVE_FEATURE_COLUMNS_KEY = 'feature_columns'


def save_model_and_feature_columns(cv: GridSearchCV, feature_columns: typing.List[str], name: str = DEFAULT_SAVE_MODEL_NAME,
                                   relative_path: str = '..', folder: str = MODELS_FOLDER, extra_data: typing.Optional[typing.Dict[str, typing.Any]] = None):
    save_dict = {SAVE_MODEL_KEY: cv.best_estimator_, SAVE_FEATURE_COLUMNS_KEY: feature_columns}
    if extra_data is not None:
        save_dict.update(extra_data)
    save_data(save_dict, folder=folder, name=name, relative_path=relative_path, data_name='fitness model')


def get_data_path(folder: str, name: str, relative_path: str = '..', delta: typing.Optional[timedelta] = None):
    date = datetime.now()
    if delta is not None:
        date -= delta
    return f'{relative_path}/{folder}/{name}_{date.strftime("%Y_%m_%d")}.pkl.gz'


def save_data(data: typing.Any, folder: str, name: str, relative_path: str = '..', log_message: bool = True,
              overwrite: bool = False, data_name: str = 'data'):
    output_path = get_data_path(folder, name, relative_path)

    i = 0
    while os.path.exists(output_path) and not overwrite:
        folder, filename = os.path.split(output_path)
        filename, period, extensions = filename.partition('.')
        if filename.endswith(f'_{i}'):
            filename = filename[:-2]

        i += 1
        filename = filename + f'_{i}'
        output_path = os.path.join(folder, filename + period + extensions)

    save_data_to_path(data, output_path, log_message=log_message, overwrite=overwrite, data_name=data_name)


def save_data_to_path(data: typing.Any, path: str, log_message: bool = True, overwrite: bool = False, data_name: str = 'data'):
    if log_message or data_name != 'data':
        logger.info(f'Saving {data_name} to {path} ...')

    if not overwrite and os.path.exists(path):
        raise FileExistsError(f'File {path} already exists.')

    open_method = gzip.open if path.endswith('.gz') else open

    with open_method(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_and_feature_columns(date_and_id: str, name: str = DEFAULT_SAVE_MODEL_NAME,
                                   relative_path: str = '..', folder: str = MODELS_FOLDER, return_full_dict: bool = False) -> typing.Tuple[GridSearchCV, typing.List[str]]:
    data = load_data(date_and_id, folder, name, relative_path)

    if BUG_SAVE_MODEL_KEY in data:
        return data[BUG_SAVE_MODEL_KEY], data[SAVE_FEATURE_COLUMNS_KEY]

    if return_full_dict:
        return data[SAVE_MODEL_KEY], data[SAVE_FEATURE_COLUMNS_KEY], data  # type: ignore

    return data[SAVE_MODEL_KEY], data[SAVE_FEATURE_COLUMNS_KEY]


def load_data(date_and_id: str, folder: str, name: str, relative_path: str = '..'):
    if date_and_id:
        if date_and_id.startswith(name):
            name = date_and_id
        else:
            name = f'{name}_{date_and_id}'
    return load_data_from_path(f'{relative_path}/{folder}/{name}.pkl.gz')


def load_data_from_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'No data file found at {path}')

    open_method = gzip.open if path.endswith('.gz') else open
    with open_method(path, 'rb') as f:
        data = pickle.load(f)

    return data


DEFAULT_RANDOM_SEED = 33
DEFAULT_TRAINING_PROP = 0.8


def train_test_split_by_game_name(df: pd.DataFrame, training_prop: float = DEFAULT_TRAINING_PROP,
    random_seed: int = DEFAULT_RANDOM_SEED, positive_column: str = 'real', positive_value: typing.Any = True,
    print_test_game_names: bool = False):

    real_game_names = df[df[positive_column] == positive_value].original_game_name.unique()

    train_game_names, test_game_names = train_test_split(real_game_names, train_size=training_prop, random_state=random_seed)
    if print_test_game_names:
        print(test_game_names)
    
    train_df = df[df.game_name.isin(train_game_names) | df.original_game_name.isin(train_game_names)]
    test_df = df[df.game_name.isin(test_game_names) | df.original_game_name.isin(test_game_names)]
    return train_df, test_df


def df_to_tensor(df: pd.DataFrame, feature_columns: typing.List[str],
    positive_column: str = 'real', positive_value: typing.Any = 1, ignore_original_game: bool = False):

    if df[positive_column].any():
        if ignore_original_game:
            positives = df.loc[df[positive_column] == positive_value, feature_columns].to_numpy()
            positives = np.expand_dims(positives, axis=1)
            negatives = df.loc[df[positive_column] != positive_value, feature_columns].to_numpy()
            n_positives = positives.shape[0]
            n_negatives_per_positive = negatives.shape[0] // n_positives
            negatives = negatives[:n_positives * n_negatives_per_positive]

            return torch.tensor(
                np.concatenate([positives, negatives.reshape(n_positives, n_negatives_per_positive, -1)], axis=1),
                dtype=torch.float
            )

        else:
            return torch.tensor(
                np.stack([
                    np.concatenate((
                        df.loc[df[positive_column] & (df.original_game_name == game_name), feature_columns].to_numpy(),
                        df.loc[(~df[positive_column]) & (df.original_game_name == game_name), feature_columns].to_numpy()
                    ))
                    for game_name
                    in df[df[positive_column] == positive_value].original_game_name.unique()
                ]),
                dtype=torch.float
            )

    else:
        return torch.tensor(df.loc[:, feature_columns].to_numpy(), dtype=torch.float)


@dataclass
class ConstrativeTrainingData:
    positive_samples: torch.Tensor
    negative_samples: torch.Tensor

    def __init_(self, positive_samples: torch.Tensor, negative_samples: typing.Union[torch.Tensor, typing.List[torch.Tensor]]):
        self.positive_samples = positive_samples
        if not isinstance(negative_samples, torch.Tensor):
            negative_samples = torch.cat(negative_samples, dim=0)
        self.negative_samples = negative_samples

    def split_train_test(self, training_prop: float = DEFAULT_TRAINING_PROP, random_seed: int = DEFAULT_RANDOM_SEED):
        positive_train, positive_test = train_test_split(self.positive_samples, train_size=training_prop, random_state=random_seed)
        negative_train, negative_test = train_test_split(self.negative_samples, train_size=training_prop, random_state=random_seed)
        return ConstrativeTrainingData(positive_train, negative_train), ConstrativeTrainingData(positive_test, negative_test)  # type: ignore


def make_init_weight_function(bias: float = 0.01):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(bias)

    return init_weights


class CustomSklearnScaler:
    def __init__(self, passthrough: bool = False):
        self.passthrough = passthrough
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        if self.passthrough:
            return self

        if X.ndim != 3:
            raise ValueError('X must be 3D')

        self.mean = X.mean(axis=(0, 1))
        self.std = X.std(axis=(0, 1))
        self.std[torch.isclose(self.std, torch.zeros_like(self.std))] = 1
        return self

    def transform(self, X, y=None):
        if self.passthrough:
            return X
        return (X - self.mean) / self.std

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        return [f'x{i}' for i in range(self.mean.shape[0])]  # type: ignore

    def set_params(self, **params):
        if params:
            if 'passthrough' in params:
                self.passthrough = params['passthrough']

        return self

    def get_params(self, deep=True):
        return dict(passthrough=self.passthrough)


class SklearnContrastiveTrainingDataWrapper:
    def __init__(self):
        self._eval = False

    def train(self, mode=True):
        self._eval = not mode

    def eval(self):
        self._eval = True

    def fit(self, X, y=None):
        return self

    def transform(self, X: typing.Union[torch.Tensor, typing.Sequence[torch.Tensor]], y=None):
        if self._eval:
            return X

        if isinstance(X, torch.Tensor):
            X = [X]

        n_features_per_tensor = [t.shape[-1] for t in X]
        if not all(n == n_features_per_tensor[0] for n in n_features_per_tensor):
            raise ValueError('All tensors must have the same number of features.')

        positives = []
        negatives = []

        for t in X:
            if t.ndim == 3:
                positives.append(t[:, 0, :])
                negatives.append(t[:, 1:, :].reshape(-1, t.shape[-1]))

            elif t.ndim == 2:
                negatives.append(t)

        return ConstrativeTrainingData(torch.cat(positives), torch.cat(negatives))


class FitnessEnergyModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: typing.Optional[int] = None,
        hidden_activation: typing.Callable = torch.relu,
        output_activation: typing.Optional[typing.Callable] = None,
        output_scaling: float = 1.0,
        n_outputs: int = 1):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        if output_activation is None:
            output_activation = nn.Identity()
        self.output_activation = output_activation
        self.output_scaling = output_scaling

        if hidden_size is None:
            self.fc1 = nn.Linear(self.n_features, self.n_outputs)
            self.hidden_activation = None

        else:
            self.fc1 = nn.Linear(self.n_features, hidden_size)
            self.fc2 = nn.Linear(hidden_size, self.n_outputs)
            self.hidden_activation = hidden_activation

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
        self.__dict__.update(state)
        if not hasattr(self, 'output_scaling'):
            self.output_scaling = 1.0

    def forward(self, x, activate: bool = True):
        x = self.fc1(x)

        if self.hidden_activation is not None:
            x = self.hidden_activation(x)
            x = self.fc2(x)

        x = self.output_scaling * x

        if self.n_outputs == 1 and activate and self.output_activation is not None:
            x = self.output_activation(x)

        return x


def _reduce(X: torch.Tensor, reduction: str, dim: typing.Optional[int] = None):
    if reduction == 'mean':
        if dim is None:
            return X.mean()
        return X.mean(dim=dim)
    elif reduction == 'sum':
        if dim is None:
            return X.sum()
        return X.sum(dim=dim)
    elif reduction.lower() == 'none':
        return X
    else:
        raise ValueError(f'Invalid reduction: {reduction}')


def fitness_nce_loss(scores: torch.Tensor, negative_score_reduction: str = 'sum', reduction: str = 'mean'):
    positive_scores = torch.log(scores[:, 0])
    negative_scores = _reduce(torch.log(1 - scores[:, 1:]), negative_score_reduction, dim=1)
    return _reduce(-(positive_scores + negative_scores), reduction)


def fitness_hinge_loss(scores: torch.Tensor, margin: float = 1.0, negative_score_reduction: str = 'none', reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    negative_scores = _reduce(scores[:, 1:], negative_score_reduction, dim=1)
    if negative_score_reduction == 'none':
        positive_scores = positive_scores.unsqueeze(-1)
    return _reduce(torch.relu(positive_scores + margin - negative_scores), reduction)


def fitness_hinge_loss_with_cross_example(scores: torch.Tensor, margin: float = 1.0, alpha: float = 0.5,
    negative_score_reduction: str = 'none', reduction: str = 'mean'):
    hinge = fitness_hinge_loss(scores, margin, negative_score_reduction, reduction)

    positive_scores = scores[:, 0, None]
    negative_scores = scores[:, 1:]
    cross_example_loss = _reduce(torch.relu(positive_scores + margin - negative_scores), reduction)

    return alpha * hinge + (1 - alpha) * cross_example_loss


def fitness_log_loss(scores: torch.Tensor, negative_score_reduction: str = 'none', reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    # negative_scores = scores[:, 1:].sum(dim=1)
    negative_scores = _reduce(scores[:, 1:], negative_score_reduction, dim=1)
    if negative_score_reduction == 'none':
        positive_scores = positive_scores.unsqueeze(-1)
    return _reduce(torch.log(1 + torch.exp(positive_scores - negative_scores)), reduction)


def fitness_weird_log_loss(scores: torch.Tensor, positive_margin: float = 2.0, margin: float = 4.0, reduction: str = 'mean',  negative_score_reduction: str = 'none'):
    positive_scores = scores[:, 0]
    negative_scores = scores[:, 1:]
    if negative_score_reduction != 'none':
        negative_scores = _reduce(negative_scores, negative_score_reduction, dim=1)
    return _reduce(torch.exp(positive_scores - positive_margin), reduction) + _reduce(torch.relu(margin - torch.log(negative_scores)), reduction)


def fitness_square_square_loss(scores: torch.Tensor, margin: float = 1.0, negative_score_reduction: str = 'none', reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    # negative_scores = scores[:, 1:].sum(dim=1)
    negative_scores = _reduce(scores[:, 1:], negative_score_reduction, dim=1)
    if negative_score_reduction == 'none':
        return fitness_square_square_loss_positive_negative_split(positive_scores, negative_scores, margin, reduction)

    return _reduce(positive_scores.pow(2) + torch.relu(margin - negative_scores).pow(2), reduction)


def fitness_square_square_loss_positive_negative_split(positive_scores: torch.Tensor, negative_scores: torch.Tensor, margin: float = 1.0, reduction: str = 'mean', negative_score_reduction: typing.Optional[str] = None):
    return _reduce(positive_scores.pow(2), reduction) + _reduce(torch.relu(margin - negative_scores).pow(2), reduction)


def fitness_softmin_loss(scores: torch.Tensor, beta: float = 1.0, negative_score_reduction: typing.Optional[str] = None, reduction: str = 'mean'):
    return nn.functional.cross_entropy(
        - beta * scores,
        torch.zeros((scores.shape[0], 1), dtype=torch.long, device=scores.device),
        reduction=reduction)


def fitness_softmin_loss_positive_to_all_negatives(scores: torch.Tensor, beta: float = 1.0, negative_score_reduction: typing.Optional[str] = None, reduction: str = 'mean'):
    positive_scores = scores[:, 0].unsqueeze(1)
    negative_scores = scores[:, 1:]
    return _inner_softmin_loss_positive_to_all_negatives(positive_scores, negative_scores, beta, negative_score_reduction, reduction)


def _inner_softmin_loss_positive_to_all_negatives(positive_scores: torch.Tensor, negative_scores: torch.Tensor, beta: float = 1.0, negative_score_reduction: typing.Optional[str] = None, reduction: str = 'mean'):
    loss = 0
    n_positives = positive_scores.shape[0]
    for i in range(n_positives):
        rolled_scores = torch.cat([torch.roll(positive_scores, i, 0), negative_scores], 1)
        loss += fitness_softmin_loss(rolled_scores, beta, negative_score_reduction, reduction)

    if reduction == 'mean':
        return loss / n_positives

    return loss


def fitness_sofmin_loss_positive_negative_split(positive_scores: torch.Tensor, negative_scores: torch.Tensor,
                                                beta: float = 1.0, reduction: str = 'mean', negative_score_reduction: typing.Optional[str] = None):
    positive_scores, negative_scores = _align_positive_and_negative_scores(positive_scores, negative_scores)

    return nn.functional.cross_entropy(
        - beta * torch.cat([positive_scores, negative_scores], dim=1),
        torch.zeros((positive_scores.shape[0], 1), dtype=torch.long, device=positive_scores.device),
        reduction=reduction)


def _align_positive_and_negative_scores(positive_scores, negative_scores):
    n_positive_scores = positive_scores.shape[0]
    n_negataive_scores = negative_scores.shape[0]
    if n_positive_scores != n_negataive_scores:
        negative_scores_per_positive = n_negataive_scores // n_positive_scores
        negative_scores = negative_scores[:int(n_positive_scores * negative_scores_per_positive)].reshape(n_positive_scores, negative_scores_per_positive, -1)

    if positive_scores.ndim < negative_scores.ndim:
        positive_scores = positive_scores.unsqueeze(1)
    return positive_scores,negative_scores


def fitness_sofmin_loss_positive_negative_split_positive_to_all_negatives(positive_scores: torch.Tensor, negative_scores: torch.Tensor,
                                                                         beta: float = 1.0, reduction: str = 'mean', negative_score_reduction: typing.Optional[str] = None):
    positive_scores, negative_scores = _align_positive_and_negative_scores(positive_scores, negative_scores)
    return _inner_softmin_loss_positive_to_all_negatives(positive_scores, negative_scores, beta, negative_score_reduction, reduction)


def fitness_softmin_hybrid_loss(scores: torch.Tensor, margin: float = 1.0, beta: float = 1.0, reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    negative_scores = scores[:, 1:]
    negative_scores_softmin = nn.functional.softmin(beta * negative_scores, dim=1)
    effective_negative_scores = torch.einsum('bij, bjk -> bik', negative_scores.squeeze(-1).unsqueeze(1), negative_scores_softmin).squeeze(-1).squeeze()
    return _reduce(torch.relu(positive_scores + margin - effective_negative_scores), reduction)
    # return _reduce(torch.log(1 + torch.exp(positive_scores - negative_scores)) + torch.relu(margin - negative_scores), reduction)


def energy_of_negative_at_quantile(positive_scores: torch.Tensor, negative_scores: torch.Tensor, quantile: float = 0.01):
    n_negatives = torch.numel(negative_scores)
    n_negatives_to_keep = int(n_negatives * quantile)
    negative_scores, _ = torch.topk(negative_scores.ravel(), n_negatives_to_keep, largest=False)
    return negative_scores.max()



DEFAULT_MODEL_KWARGS = {
    'n_features': None,
    'hidden_size': None,
    'hidden_activation': torch.relu,
    'n_outputs': 1,
    'output_activation': None,
    'output_scaling': 1.0,
}

DEFAULT_TRAIN_KWARGS = {
    'weight_decay': 0.0,
    'lr': 1e-2,
    'loss_function': fitness_nce_loss,
    'should_print': False,
    'should_tqdm': False,
    'print_interval': 10,
    'n_epochs': 1000,
    'patience_epochs': 20,
    'patience_threshold': 0.01,
    'batch_size': 8,
    'k': 4,
    'device': 'cpu',
    'dataset_energy_beta': 1.0,
    'shuffle_negatives': False,
    'shuffle_validation_negatives': None,
    'split_validation_from_train': False,
    'evaluate_opposite_shuffle_mode': False,
    'full_dataset_on_device': False,
    'regularizer': None,
    'regularization_weight': 0.0,
    'use_lr_scheduler': False,
    'lr_scheduler_class': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'lr_scheduler_mode': 'min',
    'lr_scheduler_factor': 0.5,
    'lr_scheduler_patience': None,
    'lr_scheduler_threshold': None,
    'lr_scheduler_threshold_mode': 'abs',
    'lr_scheduler_verbose': False,
    'random_seed': 33,
}


LOSS_FUNCTION_KAWRG_KEYS = ['margin', 'alpha', 'beta', 'negative_score_reduction', 'reduction']
DEFAULT_TRAIN_KWARGS.update({k: None for k in LOSS_FUNCTION_KAWRG_KEYS})

FITNESS_WRAPPER_KWARG_KEYS = ['bias_init_margin_ratio',]
DEFAULT_TRAIN_KWARGS.update({k: None for k in FITNESS_WRAPPER_KWARG_KEYS})

class SklearnFitnessWrapper:
    def __init__(self,
        model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        train_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        loss_function_kwarg_keys: typing.Sequence[str] = LOSS_FUNCTION_KAWRG_KEYS,
        fitness_wrapper_kwarg_keys: typing.Sequence[str] = FITNESS_WRAPPER_KWARG_KEYS,
        **params):

        self.model_kwargs = copy.deepcopy(DEFAULT_MODEL_KWARGS)
        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)

        self.train_kwargs = copy.deepcopy(DEFAULT_TRAIN_KWARGS)
        if train_kwargs is not None:
            self.train_kwargs.update(train_kwargs)

        self.loss_function_kwargs = {}
        self.loss_function_kwarg_keys = loss_function_kwarg_keys

        self.fitness_wrapper_kwargs = {}
        self.fitness_wrapper_kwarg_keys = fitness_wrapper_kwarg_keys

        self.losses = defaultdict(list)  # type: ignore
        self.init_model = True

        self.set_params(**params)

    def get_params(self, deep: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            **self.model_kwargs,
            **self.train_kwargs,
        }

    def set_params(self, **params) -> 'SklearnFitnessWrapper':
        for key, value in params.items():
            if key in self.model_kwargs:
                self.model_kwargs[key] = value
            elif key in self.train_kwargs:
                self.train_kwargs[key] = value
            else:
                raise ValueError(f'Unknown parameter {key}')

        return self

    def _init_model_and_train_kwargs(self):
        torch.manual_seed(self.train_kwargs['random_seed'])
        train_kwarg_keys = list(self.train_kwargs.keys())
        for key in train_kwarg_keys:
            if key in self.loss_function_kwarg_keys:
                value = self.train_kwargs.pop(key)
                if value is not None:
                    self.loss_function_kwargs[key] = value

            elif key in self.fitness_wrapper_kwarg_keys:
                value = self.train_kwargs.pop(key)
                if value is not None:
                    self.fitness_wrapper_kwargs[key] = value

        self.model = FitnessEnergyModel(**self.model_kwargs)
        bias_init_margin_ratio = self.fitness_wrapper_kwargs.get('bias_init_margin_ratio', 0)
        if 'margin' in self.train_kwargs:
            init_weights = make_init_weight_function(self.train_kwargs['margin'] * bias_init_margin_ratio)
        else:
            init_weights = make_init_weight_function(bias_init_margin_ratio)

        self.model.apply(init_weights)

        self.train_kwargs['loss_function_kwargs'] = self.loss_function_kwargs

    def fit(self, X, y=None) -> 'SklearnFitnessWrapper':
        if self.init_model:
            self._init_model_and_train_kwargs()
        if isinstance(X, ConstrativeTrainingData):
            self.model, losses = train_and_validate_model_weighted_sampling(self.model, X, **self.train_kwargs)
        else:
            if isinstance(X, tuple):
                X_train, X_val = X
            else:
                X_train = X
                X_val = None

            self.model, losses = train_and_validate_model(self.model, X_train, X_val, **self.train_kwargs)

        for key, value in losses.items():
            self.losses[key].extend(value) # type: ignore

        return self

    # def fit_with_weighted_negative_sampling(
    #     self, train_data: ConstrativeTrainingData,
    #     val_data: typing.Optional[ConstrativeTrainingData] = None,
    #     init_model: bool = True) -> 'SklearnFitnessWrapper':

    #     if init_model:
    #         self._init_model_and_train_kwargs()
    #     self.model, self.train_losses, self.val_losses = train_and_validate_model_weighted_sampling(self.model, train_data, val_data, **self.train_kwargs)
    #     return self

    def transform(self, X, y=None) -> torch.Tensor:
        return self.model(X)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        if self.model is not None:
            return self.model(*args, **kwargs)

        return torch.empty(0)



ModelClasses = typing.Union[nn.Module, SklearnFitnessWrapper, Pipeline]



def _score_samples(model: ModelClasses, X: torch.Tensor, y: typing.Optional[torch.Tensor],
    device: str = 'cpu', separate_positive_negative: bool = True) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:

    with torch.no_grad():
        X = X.to(device)

        if isinstance(model, Pipeline):
            model.named_steps['fitness'].model.to(device)
            model.named_steps['fitness'].model.eval()
            if 'wrapper' in model.named_steps:
                model.named_steps['wrapper'].eval()
            scores = model.transform(X)
            if 'wrapper' in model.named_steps:
                model.named_steps['wrapper'].train()

        elif isinstance(model, SklearnFitnessWrapper):
            model.model.to(device)
            model.model.eval()
            scores = model.transform(X)

        else:
            model.to(device)
            model.eval()
            scores = model(X, activate=False)

        scores = scores.detach().cpu()

    if separate_positive_negative:
        positive_scores = scores[:, 0]
        negative_scores = scores[:, 1:]
        return positive_scores.detach(), negative_scores.detach()

    return scores.detach()


# def evaluate_fitness(model: ModelClasses, X: torch.Tensor, y: typing.Optional[torch.Tensor] = None,
#     score_sign: int = 1):
#     positive_scores, negative_scores = _score_samples(model, X, y)

#     game_average_scores = (positive_scores - negative_scores.mean(dim=1)) * score_sign
#     return game_average_scores.mean().item()


# def evaluate_fitness_flipped_sign(model: ModelClasses,
#     X: torch.Tensor, y=None):
#     return _score_samples(model, X, y, score_sign=-1)

def _average_precision_score(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    n_positives = positive_scores.shape[0]
    negative_scores = negative_scores.reshape(-1)
    n_negatives = negative_scores.shape[0]
    labels = np.concatenate([np.ones(n_positives), np.zeros(n_negatives)])
    scores = np.concatenate([positive_scores, negative_scores]) * -1  # flipping the signs of the energies  # type: ignore
    return average_precision_score(labels, scores, average=None)  # type: ignore


def _ecdf(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    ecdf = ECDF(np.concatenate([positive_scores, negative_scores.reshape(-1)]))
    positive_mean_quantile = ecdf(positive_scores).mean()
    return -positive_mean_quantile


def evaluate_fitness_overall_ecdf_separate_tensors(model: ModelClasses,
    positives: torch.Tensor, negatives: torch.Tensor) -> float:
    positive_scores = _score_samples(model, positives, None, separate_positive_negative=False).squeeze().cpu().numpy() # type: ignore
    negative_scores = _score_samples(model, negatives, None, separate_positive_negative=False).squeeze().cpu().numpy() # type: ignore
    return _ecdf(positive_scores, negative_scores)


class ContrastiveDataWrapper:
    def __init__(self, score_function: typing.Callable):
        self.score_function = score_function

    def __call__(self, model: ModelClasses, X: typing.Union[torch.Tensor, ConstrativeTrainingData], *args, **kwargs):
        if isinstance(X, ConstrativeTrainingData):
            X = torch.cat([X.positive_samples, X.negative_samples.reshape(X.positive_samples.shape[0], -1, X.positive_samples.shape[1])], dim=0)

        return self.score_function(model, X, *args, **kwargs)


def contrastive_data_wrapper_to_tensor(score_function: typing.Callable):
    return ContrastiveDataWrapper(score_function)


@contrastive_data_wrapper_to_tensor
def evaluate_fitness_average_precision_score(model: ModelClasses,
    X: torch.Tensor, y=None) -> float:
    positive_scores, negative_scores = _score_samples(model, X, y)
    positive_scores = positive_scores.squeeze().cpu().numpy()
    negative_scores = negative_scores.squeeze().cpu().numpy()
    return _average_precision_score(positive_scores, negative_scores)


@contrastive_data_wrapper_to_tensor
def evaluate_fitness_overall_ecdf(model: ModelClasses,
    X: torch.Tensor, y=None) -> float:
    positive_scores, negative_scores = _score_samples(model, X, y)
    positive_scores = positive_scores.squeeze().cpu().numpy()
    negative_scores = negative_scores.squeeze().cpu().numpy()
    return _ecdf(positive_scores, negative_scores)


@contrastive_data_wrapper_to_tensor
def evaluate_fitness_single_game_rank(model: ModelClasses, X: torch.Tensor, y=None) -> float:
    positive_scores, negative_scores = _score_samples(model, X, y)
    single_game_rank = (positive_scores[:, None] < negative_scores).float().mean(axis=1)  # type: ignore
    return single_game_rank.mean().item()


@contrastive_data_wrapper_to_tensor
def evaluate_fitness_single_game_min_rank(model: ModelClasses, X: torch.Tensor, y=None) -> float:
    positive_scores, negative_scores = _score_samples(model, X, y)
    single_game_rank = (positive_scores[:, None] < negative_scores).float().mean(axis=1)  # type: ignore
    return single_game_rank.min().item()


class LossFunctionMetricWrapper:
    def __init__(self, loss_function: typing.Callable[..., torch.Tensor],
                                 kwargs: typing.Optional[dict] = None, positive_negative_split: bool = False,
                                 flip_sign: bool = True):

        self.loss_function = loss_function
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        self.positive_negative_split = positive_negative_split
        self.flip_sign = flip_sign

    def __call__(self, model: ModelClasses, X: torch.Tensor, y=None) -> float:
        positive_scores, negative_scores = _score_samples(model, X, y)
        if self.positive_negative_split:
            score = self.loss_function(positive_scores, negative_scores, **self.kwargs)  # type: ignore

        else:
            n_positive_scores = positive_scores.shape[0]
            n_negataive_scores = negative_scores.shape[0]
            if n_positive_scores != n_negataive_scores:
                negative_scores_per_positive = n_negataive_scores // n_positive_scores
                negative_scores = negative_scores[:int(n_positive_scores * negative_scores_per_positive)].reshape(n_positive_scores, negative_scores_per_positive, -1)

            if positive_scores.ndim < negative_scores.ndim:
                positive_scores = positive_scores.unsqueeze(1)
            scores = torch.cat([positive_scores, negative_scores], dim=1) # type: ignore
            score = self.loss_function(scores, **self.kwargs)  # type: ignore

        if isinstance(score, torch.Tensor):
            score = score.item()

        return -score if self.flip_sign else score


@contrastive_data_wrapper_to_tensor
def wrap_loss_function_to_metric(loss_function: typing.Callable[..., torch.Tensor],
                                 kwargs: typing.Optional[dict] = None, positive_negative_split: bool = False,
                                 flip_sign: bool = True) -> typing.Callable[[ModelClasses, torch.Tensor, typing.Optional[torch.Tensor]], float]:
    return LossFunctionMetricWrapper(loss_function, kwargs, positive_negative_split, flip_sign)


class MultipleScoringWrapper:
    def __init__(self,
                 evaluators: typing.Sequence[typing.Callable[[ModelClasses, typing.Union[torch.Tensor, ConstrativeTrainingData], typing.Optional[torch.Tensor]], float]],
                 names: typing.Sequence[str]):

        self.evaluators = evaluators
        self.names = names

    def __call__(self, model: ModelClasses, X: typing.Union[torch.Tensor, ConstrativeTrainingData], y=None, return_all=False):
        return {name: evaluator(model, X, y) for name, evaluator in zip(self.names, self.evaluators)}


def build_multiple_scoring_function(
    evaluators: typing.Sequence[typing.Callable[[ModelClasses, typing.Union[torch.Tensor, ConstrativeTrainingData], typing.Optional[torch.Tensor]], float]],
    names: typing.Sequence[str]
    ) -> typing.Callable[[ModelClasses, typing.Union[torch.Tensor, ConstrativeTrainingData], typing.Optional[torch.Tensor]], typing.Dict[str, float]]:
    # def _evaluate_fitness_multiple(model: ModelClasses, X: typing.Union[torch.Tensor, ConstrativeTrainingData], y=None, return_all=False):
    #     return {name: evaluator(model, X, y) for name, evaluator in zip(names, evaluators)}

    # _evaluate_fitness_multiple.names = names

    return MultipleScoringWrapper(evaluators, names)


class ModelRegularizer:
    def __init__(self, ord: int, threshold: typing.Optional[float] = None):
        self.ord = ord
        self.threshold = threshold

    def __call__(self, model: ModelClasses) -> torch.Tensor:
        if isinstance(model, Pipeline):
            model = model.named_steps['fitness']

        if isinstance(model, SklearnFitnessWrapper):
            model = model.model

        if self.threshold is not None and self.ord == 0:
            weight = torch.zeros_like(model.fc1.weight)
            weight[model.fc1.weight.abs() > self.threshold] = 1.0
            return torch.linalg.vector_norm(weight, ord=self.ord)

        return torch.linalg.vector_norm(model.fc1.weight, ord=self.ord)


default_multiple_scoring = build_multiple_scoring_function(
    [
        wrap_loss_function_to_metric(fitness_sofmin_loss_positive_negative_split, dict(beta=1.0), True),  # type: ignore
        evaluate_fitness_overall_ecdf,
        evaluate_fitness_single_game_rank,
        evaluate_fitness_single_game_min_rank,
        wrap_loss_function_to_metric(energy_of_negative_at_quantile, dict(quantile=0.01), True),  # type: ignore
        wrap_loss_function_to_metric(energy_of_negative_at_quantile, dict(quantile=0.05), True),  # type: ignore
        evaluate_fitness_average_precision_score,
    ],
    [
        'loss',
        'overall_ecdf',
        'single_game_rank',
        'single_game_min_rank',
        'energy_of_negative@1%',
        'energy_of_negative@5%',
        'average_precision_score',
    ],
)


DEFAULT_INITIAL_ENERGY = 1.0
DEFAULT_ENERGY_BETA = 1.0
DEFAULT_MAX_ENERGY = 50.0

class EnergyRecencyWeightedDataset(IterableDataset):
    current_epoch: int
    data: ConstrativeTrainingData
    energy_beta: float
    initial_energy: float
    k: int
    n_positives: int
    negative_energies: torch.Tensor
    negative_last_sampled: torch.Tensor
    positive_order: torch.Tensor

    def __init__(self, data: ConstrativeTrainingData, k: int = 4,
                 energy_beta: float = DEFAULT_ENERGY_BETA, initial_energy: float = DEFAULT_INITIAL_ENERGY,
                 max_energy: float = DEFAULT_MAX_ENERGY, device: str = 'cpu'):
        self.data = data
        self.k = k
        self.energy_beta = energy_beta
        self.initial_energy = initial_energy
        self.max_energy = max_energy
        self.device = device

        self.n_positives = len(data.positive_samples)
        self.negative_energies = torch.ones(len(data.negative_samples), device=self.device) * initial_energy
        self.negative_last_sampled = torch.zeros(len(data.negative_samples), device=self.device)
        self.current_epoch = 0

        self.latest_negative_indices_yielded = []

    def _new_epoch(self):
        self.current_epoch += 1
        epochs_not_sampled = self.current_epoch - self.negative_last_sampled
        unnormalized_logprobs = epochs_not_sampled - (self.negative_energies * self.energy_beta)
        unnormalized_logprobs[~torch.isfinite(unnormalized_logprobs)] = -torch.inf
        shifted_logprobs = unnormalized_logprobs - unnormalized_logprobs.max()
        probs = torch.exp(shifted_logprobs)
        probs = probs / probs.sum()

        negative_indices = torch.multinomial(probs, self.n_positives * self.k, replacement=False)

        # negative_index_energy_mean = self.negative_energies[negative_indices].mean()
        # negative_index_energy_std = self.negative_energies[negative_indices].std()
        # negative_index_last_sampled_mean = self.negative_last_sampled[negative_indices].mean()
        # negative_index_last_sampled_std = self.negative_last_sampled[negative_indices].std()
        # negative_index_unnormalized_logprobs_mean = unnormalized_logprobs[negative_indices].mean()
        # negative_index_unnormalized_logprobs_std = unnormalized_logprobs[negative_indices].std()

        # print(f"Energy mean: {negative_index_energy_mean:.3f} +/- {negative_index_energy_std:.3f}  | Last sampled mean: {negative_index_last_sampled_mean:.3f} +/- {negative_index_last_sampled_std:.3f} | Unnormalized logprobs mean: {negative_index_unnormalized_logprobs_mean:.3f} +/- {negative_index_unnormalized_logprobs_std:.3f}")

        self.negative_last_sampled[negative_indices] = self.current_epoch
        self.negative_indices_per_positive = negative_indices.view(self.n_positives, self.k)

        self.positive_order = torch.randperm(self.n_positives)

    def __iter__(self):
        self._new_epoch()
        for positive_index in self.positive_order:
            negative_indices = self.negative_indices_per_positive[positive_index]
            self.latest_negative_indices_yielded.append(negative_indices)
            yield torch.cat((self.data.positive_samples[positive_index].unsqueeze(0), self.data.negative_samples[negative_indices]))

    def update_negative_energies(self, negative_energies: torch.Tensor):
        negative_indices = torch.stack(self.latest_negative_indices_yielded).squeeze()
        if negative_energies.ndim > 1:
            negative_energies = negative_energies.squeeze(0)

        if negative_indices.shape != negative_energies.shape:
            raise ValueError(f"Negative indices shape {negative_indices.shape} does not match negative energies shape {negative_energies.shape}")

        negative_energies[negative_energies > self.max_energy] = self.max_energy
        negative_energies[~torch.isfinite(negative_energies)] = self.max_energy

        self.negative_energies[negative_indices] = negative_energies
        self.latest_negative_indices_yielded = []


def train_and_validate_model_weighted_sampling(
    model: nn.Module,
    train_data: ConstrativeTrainingData,
    val_data: typing.Optional[ConstrativeTrainingData] = None,
    loss_function: typing.Callable = fitness_square_square_loss,
    loss_function_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    val_loss_function: typing.Callable = fitness_sofmin_loss_positive_negative_split,
    optimizer_class: typing.Callable = torch.optim.SGD,
    n_epochs: int = 1000, lr: float = 0.01, weight_decay: float = 0.0,
    should_print: bool = True, should_tqdm: bool = False,
    should_print_weights: bool = False, print_interval: int = 10,
    patience_epochs: int = 5, patience_threshold: float = 0.01,
    batch_size: int = 8, k: int = 4,
    dataset_energy_beta: float = DEFAULT_ENERGY_BETA,
    dataset_initial_energy: float = DEFAULT_INITIAL_ENERGY,
    split_validation_from_train: bool = False,
    regularizer: typing.Optional[typing.Callable[[nn.Module], torch.Tensor]] = None, regularization_weight: float = 0.0,
    use_lr_scheduler: bool = False, lr_scheduler_class: typing.Callable = torch.optim.lr_scheduler.ReduceLROnPlateau,
    lr_scheduler_mode: str = 'min', lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: typing.Optional[int] = None, lr_scheduler_threshold: typing.Optional[float] = None,
    lr_scheduler_threshold_mode: str = 'abs', lr_scheduler_verbose: bool = False,
    num_workers: int = 0, device: str = 'cpu', random_seed: int = 33, **kwargs,
    ) -> typing.Tuple[nn.Module, typing.Dict[str, typing.List[float]]]:

    if loss_function_kwargs is None:
        loss_function_kwargs = {}

    torch.manual_seed(random_seed)
    model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_lr_scheduler:
        if lr_scheduler_patience is None:
            lr_scheduler_patience = patience_epochs // 2
        if lr_scheduler_threshold is None:
            lr_scheduler_threshold = patience_threshold
        scheduler = lr_scheduler_class(
            optimizer, lr_scheduler_mode,
            factor=lr_scheduler_factor, patience=lr_scheduler_patience,
            threshold=lr_scheduler_threshold, threshold_mode=lr_scheduler_threshold_mode,
            verbose=lr_scheduler_verbose)

    if split_validation_from_train and val_data is None:
        train_data, val_data = train_data.split_train_test(training_prop=DEFAULT_TRAINING_PROP, random_seed=random_seed)  # type: ignore

    train_dataset = EnergyRecencyWeightedDataset(train_data, k=k, energy_beta=dataset_energy_beta, initial_energy=dataset_initial_energy, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    validate = val_data is not None

    min_loss = np.Inf
    patience_loss = np.Inf
    patience_update_epoch = 0
    best_model = model
    train_losses = []
    val_losses = []

    epoch = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_train_losses = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            X = batch.to(device)
            scores = model(X)
            train_dataset.update_negative_energies(scores[:, 1:].detach().squeeze())
            loss = loss_function(scores, **loss_function_kwargs)
            if regularizer is not None:
                loss += regularization_weight * regularizer(model)
            epoch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_val_losses = []

        if validate:
            model.eval()
            with torch.no_grad():
                val_positive_scores = model(val_data.positive_samples.to(device))
                val_negative_scores = model(val_data.negative_samples.to(device))
                val_loss = val_loss_function(val_positive_scores, val_negative_scores, **loss_function_kwargs)
                if regularizer is not None:
                    val_loss += regularization_weight * regularizer(model)
                epoch_val_losses.append(val_loss.item())

        if should_print and epoch % print_interval == 0:
            if validate:
                if should_print_weights:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | val loss {np.mean(epoch_val_losses):.4f} | weights {model.fc1.weight.data}')  # type: ignore
                else:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | val loss {np.mean(epoch_val_losses):.4f}')
            else:
                if should_print_weights:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | weights {model.fc1.weight.data}')  # type: ignore
                else:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f}')

        epoch_train_loss = np.mean(epoch_train_losses)
        train_losses.append(epoch_train_loss)
        if validate:
            epoch_loss = np.mean(epoch_val_losses)
            val_losses.append(epoch_loss)
        else:
            epoch_loss = epoch_train_loss

        if scheduler is not None:
            scheduler.step(epoch_loss)

        if epoch_loss < min_loss:
            if should_print:
                print(f'Epoch {epoch}: new best model with loss {epoch_loss:.4f}')
            min_loss = epoch_loss
            x = copy.deepcopy(model).cpu()

        if epoch_loss < patience_loss - patience_threshold:
            if should_print:
                print(f'Epoch {epoch}: updating patience loss from {patience_loss:.4f} to {epoch_loss:.4f}')
            patience_loss = epoch_loss
            patience_update_epoch = epoch

        if epoch - patience_update_epoch >= patience_epochs:
            if should_print:
                print(f'Early stopping after {epoch} epochs')
            break

    if epoch == n_epochs - 1:
        print('Training finished without early stopping')

    model = best_model.to(device)

    return model, dict(train=train_losses, val=val_losses)


class NegativeShuffleDataset(IterableDataset):
    dataset: torch.Tensor
    negatives: torch.Tensor
    positives: torch.Tensor
    shuffle_only_once: bool
    shuffled: bool

    def __init__(self, tensor, shuffle_only_once: bool = False):
        self.positives = torch.clone(tensor[:, 0, :])
        self.negatives = torch.clone(tensor[:, 1:, :])
        self.shuffle_only_once = shuffle_only_once
        self.shuffled = False

    def __len__(self):
        return self.positives.shape[0]

    def _new_epoch(self):
        if self.shuffle_only_once and self.shuffled:
            return

        self.shuffled = True
        negatives_permuation = torch.randperm(self.negatives.shape[0] * self.negatives.shape[1])
        shuffled_negatives = self.negatives.reshape(-1, self.negatives.shape[2])[negatives_permuation].reshape(self.negatives.shape)
        positives_permutation = torch.randperm(self.positives.shape[0])
        self.dataset = torch.cat((self.positives[positives_permutation].unsqueeze(1), shuffled_negatives), dim=1)

    def __iter__(self):
        self._new_epoch()
        return iter(self.dataset)


def train_and_validate_model(model: nn.Module,
    train_data: torch.Tensor,
    val_data: typing.Optional[torch.Tensor] = None,
    loss_function: typing.Callable = fitness_square_square_loss,
    loss_function_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    optimizer_class: typing.Callable = torch.optim.SGD,
    n_epochs: int = 1000, lr: float = 0.01, weight_decay: float = 0.0,
    should_print: bool = False, should_tqdm: bool = False,
    should_print_weights: bool = False, print_interval: int = 10,
    patience_epochs: int = 5, patience_threshold: float = 0.01,
    shuffle_negatives: bool = False, shuffle_validation_negatives: typing.Optional[bool] = None,
    evaluate_opposite_shuffle_mode: bool = False, split_validation_from_train: bool = False,
    regularizer: typing.Optional[typing.Callable[[nn.Module], torch.Tensor]] = None, regularization_weight: float = 0.0,
    full_dataset_on_device: bool = False,
    use_lr_scheduler: bool = False, lr_scheduler_class: typing.Callable = torch.optim.lr_scheduler.ReduceLROnPlateau,
    lr_scheduler_mode: str = 'min', lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: typing.Optional[int] = None, lr_scheduler_threshold: typing.Optional[float] = None,
    lr_scheduler_threshold_mode: str = 'abs', lr_scheduler_verbose: bool = False,
    batch_size: int = 8, k: int = 4, device: str = 'cpu', random_seed: int = 33,
    **kwargs) -> typing.Tuple[nn.Module, typing.Dict[str, typing.List[float]]]:

    if loss_function_kwargs is None:
        loss_function_kwargs = {}

    torch.manual_seed(random_seed)
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    scheduler = None
    if use_lr_scheduler:
        if lr_scheduler_patience is None:
            lr_scheduler_patience = patience_epochs // 2
        if lr_scheduler_threshold is None:
            lr_scheduler_threshold = patience_threshold
        scheduler = lr_scheduler_class(
            optimizer, lr_scheduler_mode,
            factor=lr_scheduler_factor, patience=lr_scheduler_patience,
            threshold=lr_scheduler_threshold, threshold_mode=lr_scheduler_threshold_mode,
            verbose=lr_scheduler_verbose)

    if split_validation_from_train and val_data is None:
        train_data, val_data = train_test_split(train_data, random_state=random_seed, train_size=DEFAULT_TRAINING_PROP)  # type: ignore

    if full_dataset_on_device:
        train_data = train_data.to(device)

    if shuffle_negatives:
        train_dataset = NegativeShuffleDataset(train_data)  # .to(device)
        shuffle = None
    else:
        train_dataset = TensorDataset(train_data)  # .to(device)
        shuffle = True

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    if shuffle_validation_negatives is None:
        shuffle_validation_negatives = shuffle_negatives

    validate = val_data is not None
    if validate:
        if full_dataset_on_device:
            val_data = val_data.to(device)

        if shuffle_validation_negatives:
            val_dataset = NegativeShuffleDataset(val_data, shuffle_only_once=True)  # .to(device)
        else:
            val_dataset = TensorDataset(val_data)

        val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    extra_evaluation_sets = {}
    if evaluate_opposite_shuffle_mode:
        if shuffle_negatives:
            opposite_train_dataset = TensorDataset(train_data)  # .to(device)
        else:
            opposite_train_dataset = NegativeShuffleDataset(train_data, shuffle_only_once=True)

        opposite_train_dataloader = DataLoader(opposite_train_dataset, batch_size=len(opposite_train_dataset), shuffle=False)
        extra_evaluation_sets[f'{"no_shuffle" if shuffle_negatives else "shuffle"}_train'] = opposite_train_dataloader

        if validate:
            if shuffle_validation_negatives:
                opposite_val_dataset = TensorDataset(val_data)  # type: ignore
            else:
                opposite_val_dataset = NegativeShuffleDataset(val_data, shuffle_only_once=True)

            opposite_val_dataloader = DataLoader(opposite_val_dataset, batch_size=len(opposite_val_dataset), shuffle=False)
            extra_evaluation_sets[f'{"no_shuffle" if shuffle_validation_negatives else "shuffle"}_val'] = opposite_val_dataloader

    torch.manual_seed(random_seed)

    min_loss = np.Inf
    patience_loss = np.Inf
    patience_update_epoch = 0
    best_model = model
    losses = defaultdict(list)

    if validate:
        pre_val_losses = []
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:    # type: ignore
                loss = _get_batch_loss(model, loss_function, loss_function_kwargs, k, device, batch)
                if regularizer is not None:
                    loss += regularization_weight * regularizer(model)
                pre_val_losses.append(loss.item())

        patience_loss = np.mean(pre_val_losses)
        min_loss = patience_loss

    pbar = None
    if should_tqdm:
        pbar = tqdm(total=n_epochs)

    epoch = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_train_losses = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = _get_batch_loss(model, loss_function, loss_function_kwargs, k, device, batch)
            if regularizer is not None:
                loss += regularization_weight * regularizer(model)
            epoch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_val_losses = []

        if validate:
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:    # type: ignore
                    loss = _get_batch_loss(model, loss_function, loss_function_kwargs, k, device, batch)
                    if regularizer is not None:
                        loss += regularization_weight * regularizer(model)
                    epoch_val_losses.append(loss.item())


        if len(extra_evaluation_sets) > 0:
            model.eval()
            with torch.no_grad():
                for name, dataloader in extra_evaluation_sets.items():
                    epoch_eval_losses = []
                    for batch in dataloader:
                        loss = _get_batch_loss(model, loss_function, loss_function_kwargs, k, device, batch)
                        if regularizer is not None:
                            loss += regularization_weight * regularizer(model)
                        epoch_eval_losses.append(loss.item())

                    losses[name].append(np.mean(epoch_eval_losses))

        epoch_train_loss = np.mean(epoch_train_losses)

        losses['train'].append(epoch_train_loss)
        if validate:
            epoch_loss = np.mean(epoch_val_losses)
            losses['val'].append(epoch_loss)
        else:
            epoch_loss = epoch_train_loss

        if scheduler is not None:
            scheduler.step(epoch_loss)

        if epoch_loss < min_loss:
            if should_print:
                print(f'Epoch {epoch}: new best model with loss {epoch_loss:.4f}')
            min_loss = epoch_loss
            best_model = copy.deepcopy(model).cpu()

        if epoch_loss < patience_loss - patience_threshold:
            if should_print:
                print(f'Epoch {epoch}: updating patience loss from {patience_loss:.4f} to {epoch_loss:.4f}')
            patience_loss = epoch_loss
            patience_update_epoch = epoch

        if pbar is not None:
            pbar.update(1)
            postfix = dict()
            if validate:
                postfix['train_loss'] = epoch_train_loss
                postfix['val_loss'] = epoch_loss
            else:
                postfix['loss'] = epoch_loss

            postfix['min_loss'] = min_loss
            postfix['patience_update_epoch'] = patience_update_epoch
            pbar.set_postfix(postfix)

        if epoch - patience_update_epoch >= patience_epochs:
            if should_print:
                print(f'Early stopping after {epoch} epochs')
            break

    if epoch == n_epochs - 1:
        print('Training finished without early stopping')

    model = best_model.to(device)

    return model, losses


def _get_batch_loss(model: torch.nn.Module, loss_function: typing.Callable,
                    loss_function_kwargs: typing.Dict[str, typing.Any],
                    k: int, device: str, X: torch.Tensor) -> torch.Tensor:
    if not isinstance(X, torch.Tensor):
        X = X[0]

    if k != X.shape[1] - 1:
        negative_indices = torch.randperm(X.shape[1] - 1)[:k] + 1
        indices = torch.cat((torch.tensor([0]), negative_indices))
        X = X[:, indices]

    scores = model(X.to(device))
    loss = loss_function(scores, **loss_function_kwargs)
    return loss


def initialize_and_fit_model(
        input_data: typing.Union[pd.DataFrame, torch.Tensor, typing.Sequence[pd.DataFrame], typing.Sequence[torch.Tensor]],
        split_test_set: bool = True, split_validation_set: bool = True,
        feature_columns: typing.Optional[typing.List[str]] = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        scaler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        train_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        energy_weighted_resampling: bool = False,
        train_prop: float = DEFAULT_TRAINING_PROP,
        pipeline: typing.Optional[Pipeline] = None,
        scoring_function: typing.Optional[typing.Callable] = None,
    ) -> typing.Tuple[Pipeline, typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor], typing.Optional[torch.Tensor]], typing.Dict[str, typing.Dict[str, typing.Any]]]:

    if scaler_kwargs is None:
        scaler_kwargs = {}

    if model_kwargs is None:
        model_kwargs = {}

    if train_kwargs is None:
        train_kwargs = {}

    train_tensor, test_tensor = _input_data_to_train_test_tensors(input_data=input_data, feature_columns=feature_columns,
        split_test_set=split_test_set, random_seed=random_seed, train_prop=train_prop)

    val_tensor = None
    if split_validation_set:
        train_tensor, val_tensor = train_test_split(train_tensor, random_state=random_seed, train_size=train_prop)

    train_tensor = typing.cast(torch.Tensor, train_tensor)
    model_kwargs['n_features'] = train_tensor.shape[-1]

    if pipeline is None:
        if energy_weighted_resampling:
            pipeline = Pipeline(steps=[('wrapper', SklearnContrastiveTrainingDataWrapper()), ('fitness', SklearnFitnessWrapper(model_kwargs=model_kwargs, train_kwargs=train_kwargs)),])

        else:
            pipeline = Pipeline(steps=[('scaler', CustomSklearnScaler(**scaler_kwargs)), ('fitness', SklearnFitnessWrapper(model_kwargs=model_kwargs, train_kwargs=train_kwargs))])

    if split_validation_set:
        pipeline.fit(train_tensor, val_tensor)
    else:
        pipeline.fit(train_tensor)

    train_results = evaluate_trained_model(pipeline, train_tensor, scoring_function)  # type: ignore
    val_results = None
    if split_validation_set:
        val_results = evaluate_trained_model(pipeline, val_tensor, scoring_function)  # type: ignore

    test_results = None
    combined_results = None

    if split_test_set:
        if test_tensor is None:
            raise ValueError(f'Encoutered None test tensor with split_test_set=True')
        test_tensor = typing.cast(torch.Tensor, test_tensor)
        test_results = evaluate_trained_model(pipeline, test_tensor, scoring_function)  # type: ignore

        combined_tensor = torch.cat([train_tensor, val_tensor, test_tensor], dim=0) if split_validation_set else torch.cat([train_tensor, test_tensor], dim=0)    # type: ignore
        combined_results = evaluate_trained_model(pipeline, combined_tensor, scoring_function)  # type: ignore

    return pipeline, (train_tensor, val_tensor, test_tensor), dict(train=train_results, val=val_results, test=test_results, combined=combined_results)  # type: ignore


# class TQDMParameterGrid(ParameterGrid):
#     def __init__(self, param_grid: typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]]):
#         super().__init__(param_grid)

#     def __iter__(self):
#         for p in tqdm(super().__iter__(), total=len(self)):  # type: ignore
#             yield p


# class TQDMGridSearchCV(GridSearchCV):
#     def __init__(self, estimator: typing.Any, param_grid: typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]], *args, **kwargs):
#         super().__init__(estimator, param_grid, *args, **kwargs)

#     def _run_search(self, evaluate_candidates):
#         """Search all candidates in param_grid"""
#         evaluate_candidates(TQDMParameterGrid(self.param_grid))


def cross_validate(train: torch.Tensor,
    param_grid: typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]],
    scoring_function: typing.Callable = evaluate_fitness_overall_ecdf,
    scaler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    train_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    cv_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    n_folds: int = 5, energy_weighted_resampling: bool = False, verbose: int = 0) -> GridSearchCV:

    if scaler_kwargs is None:
        scaler_kwargs = {}

    if model_kwargs is None:
        model_kwargs = {}

    if train_kwargs is None:
        train_kwargs = {}

    if cv_kwargs is None:
        cv_kwargs = {}

    if 'n_jobs' not in cv_kwargs:
        cv_kwargs['n_jobs'] = -1
    if 'verbose' not in cv_kwargs:
        cv_kwargs['verbose'] = verbose

    if energy_weighted_resampling:
        pipeline = Pipeline(steps=[('wrapper', SklearnContrastiveTrainingDataWrapper()), ('fitness', SklearnFitnessWrapper(model_kwargs=model_kwargs, train_kwargs=train_kwargs)),])

    else:
        pipeline = Pipeline(steps=[('scaler', CustomSklearnScaler(**scaler_kwargs)), ('fitness', SklearnFitnessWrapper(model_kwargs=model_kwargs, train_kwargs=train_kwargs))])


    n_features = train.shape[-1]

    if isinstance(param_grid, list):
        for param_grid_dict in param_grid:
            param_grid_dict['fitness__n_features'] = [n_features]
    else:
        param_grid['fitness__n_features'] = [n_features]

    random_seed = train_kwargs['random_seed'] if 'random_seed' in train_kwargs else None

    cv = GridSearchCV(pipeline, param_grid, scoring=scoring_function,
        cv=KFold(n_folds, shuffle=True, random_state=random_seed),
        **cv_kwargs)
    return cv.fit(train, None)


def model_fitting_experiment(input_data: typing.Union[pd.DataFrame, torch.Tensor, typing.Sequence[pd.DataFrame], typing.Sequence[torch.Tensor]],
    param_grid: typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]],
    split_test_set: bool = True,
    feature_columns: typing.Optional[typing.List[str]] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    scoring_function: typing.Callable = evaluate_fitness_overall_ecdf,
    scaler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    train_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    cv_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    n_folds: int = 5, energy_weighted_resampling: bool = False,
    train_prop: float = DEFAULT_TRAINING_PROP, verbose: int = 0,
    ) -> typing.Tuple[GridSearchCV, typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]], typing.Dict[str, typing.Dict[str, float]]]:

    if scaler_kwargs is None:
        scaler_kwargs = {}

    if model_kwargs is None:
        model_kwargs = {}

    if train_kwargs is None:
        train_kwargs = {}

    cv_tensor, test_tensor = _input_data_to_train_test_tensors(input_data=input_data, feature_columns=feature_columns,
        split_test_set=split_test_set, random_seed=random_seed, train_prop=train_prop, print_test_game_names=True)

    if test_tensor is not None:
        print(f'Train tensor shape: {cv_tensor.shape} | Test tensor shape: {test_tensor.shape}')  # type: ignore
    else:
        print(f'Train tensor shape: {cv_tensor.shape}')

    cv = cross_validate(cv_tensor, param_grid,
        scoring_function=scoring_function,
        scaler_kwargs=scaler_kwargs, model_kwargs=model_kwargs,
        train_kwargs={'random_seed': random_seed, **train_kwargs},
        cv_kwargs=cv_kwargs, n_folds=n_folds,
        energy_weighted_resampling=energy_weighted_resampling, verbose=verbose)

    train_tensor = typing.cast(torch.Tensor, cv_tensor)
    best_model = typing.cast(SklearnFitnessWrapper, cv.best_estimator_)
    train_results = evaluate_trained_model(best_model, train_tensor, scoring_function)

    test_results = None
    combined_results = None

    if split_test_set:
        if test_tensor is None:
            raise ValueError(f'Encoutered None test tensor with split_test_set=True')
        test_tensor = typing.cast(torch.Tensor, test_tensor)
        test_results = evaluate_trained_model(best_model, test_tensor, scoring_function)

        combined_tensor = torch.cat([train_tensor, test_tensor], dim=0)
        combined_results = evaluate_trained_model(best_model, combined_tensor, scoring_function)

    return cv, (cv_tensor, test_tensor), dict(train=train_results, test=test_results, combined=combined_results)  # type: ignore


def _input_data_to_train_test_tensors(
    input_data: typing.Union[pd.DataFrame, torch.Tensor, typing.Sequence[pd.DataFrame], typing.Sequence[torch.Tensor]],
    feature_columns: typing.Optional[typing.List[str]],
    split_test_set: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
    train_prop: float = DEFAULT_TRAINING_PROP,
    ignore_original_game: bool = False, print_test_game_names: bool = False) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]:

    test_tensor = None

    if isinstance(input_data, pd.DataFrame):
        if feature_columns is None:
            feature_columns = [str(c) for c in input_data.columns if c not in NON_FEATURE_COLUMNS]

        if split_test_set:
            input_data, test_data = train_test_split_by_game_name(input_data, random_seed=random_seed, print_test_game_names=print_test_game_names)
            test_tensor = df_to_tensor(test_data, feature_columns, ignore_original_game=ignore_original_game)

        input_data = typing.cast(pd.DataFrame, input_data)
        train_tensor = df_to_tensor(input_data, feature_columns, ignore_original_game=ignore_original_game)

    else:
        if isinstance(input_data, (list, tuple)):
            first_input = input_data[0]
            if isinstance(first_input, pd.DataFrame):
                if feature_columns is None:
                    feature_columns = [str(c) for c in first_input.columns if c not in NON_FEATURE_COLUMNS]  # type: ignore

                input_data = [df_to_tensor(df, feature_columns) for df in input_data]   # type: ignore

            if input_data[0].ndim != 3:
                raise ValueError('If cv_data is a list or tuple, the first tensor must be 3D [n_positives] x [1 + n_negatives] x [n_features]]')

            tensors = [input_data[0]]
            n_positives = input_data[0].shape[0]

            for t in input_data[1:]:
                if t.ndim == 3:
                    if t.shape[0] != n_positives:
                        raise ValueError('If cv_data is a list or tuple, all 3D tensors must have the same number of positives')
                    tensors.append(t)

                elif t.ndim == 2:  # negatives only
                    negatives_per_positive = t.shape[0] // n_positives
                    t = t[:negatives_per_positive * n_positives].reshape(n_positives, negatives_per_positive, -1)
                    tensors.append(t)

            input_data = torch.cat(tensors, dim=1)  # type: ignore

        train_tensor = input_data
        if split_test_set:
            train_tensor, test_tensor = train_test_split(train_tensor, random_state=random_seed,
                train_size=train_prop)

    return train_tensor, test_tensor  # type: ignore


def evaluate_trained_model(model: SklearnFitnessWrapper, data_tensor: torch.Tensor,
                           scoring_function: typing.Optional[typing.Callable] = None) -> typing.Dict[str, float]:

    if scoring_function is not None:
        output = scoring_function(model, data_tensor)
        if not isinstance(output, dict):
            output = dict(score=output)

        shuffled_dataset = NegativeShuffleDataset(data_tensor)
        shuffled_dataset._new_epoch()
        shuffled_output = scoring_function(model, shuffled_dataset.dataset)

        if not isinstance(shuffled_output, dict):
            shuffled_output = dict(score=shuffled_output)

        output.update({'shuffled_' + k: v for k, v in shuffled_output.items()})
        return output

    else:
        combined_ecdf = evaluate_fitness_overall_ecdf(model, data_tensor)
        combined_game_rank = evaluate_fitness_single_game_rank(model, data_tensor)
        combined_game_min_rank = evaluate_fitness_single_game_min_rank(model, data_tensor)
        combined_mean_average_precision = evaluate_fitness_average_precision_score(model, data_tensor)
        combined_results = dict(ecdf=combined_ecdf, game_rank=combined_game_rank, game_min_rank=combined_game_min_rank, mean_average_precision=combined_mean_average_precision)

    return combined_results


def plot_energy_histogram(energy_model: typing.Union[GridSearchCV, Pipeline],
    train_tensor: torch.Tensor, test_tensor: typing.Optional[torch.Tensor] = None,
    histogram_title_base: str = 'Energy scores of all games',
    histogram_title_note: typing.Optional[str] = None,
    histogram_log_y: bool = True):

    if isinstance(energy_model, GridSearchCV):
        energy_model = energy_model.best_estimator_  # type: ignore

    energy_model = typing.cast(Pipeline, energy_model)

    if 'wrapper' in energy_model.named_steps:
        energy_model.named_steps['wrapper'].eval()
    train_positive_scores = energy_model.transform(train_tensor[:, 0, :]).detach().squeeze().numpy()  # type: ignore
    train_negative_scores = energy_model.transform(train_tensor[:, 1:, :]).detach().squeeze().numpy()  # type: ignore
    hist_scores = [train_positive_scores, train_negative_scores.flatten()]
    cm = plt.get_cmap('tab20')  # type: ignore
    colors = cm.colors[0], cm.colors[2]

    if test_tensor is not None:
        labels = ['Real (train)', 'Regrown (train)']
        test_positive_scores = energy_model.transform(test_tensor[:, 0, :]).detach().squeeze().numpy()  # type: ignore
        test_negative_scores = energy_model.transform(test_tensor[:, 1:, :]).detach().squeeze().numpy()  # type: ignore

        hist_scores.insert(1, test_positive_scores)
        hist_scores.append(test_negative_scores.flatten())
        labels.insert(1, 'Real (test)')
        labels.append('Regrown (test)')

        colors = cm.colors[:4]

    else:
        labels = ['Real', 'Regrown']

    plt.hist(hist_scores, label=labels, stacked=True, bins=100, color=colors)  # type: ignore
    if histogram_title_note is not None:
        plt.title(f'{histogram_title_base} ({histogram_title_note})')
    else:
        plt.title(histogram_title_base)

    plt.xlabel('Energy score')

    if histogram_log_y:
        plt.ylabel('log(Count)')
        plt.semilogy()
    else:
        plt.ylabel('Count')

    plt.legend(loc='best')
    plt.show()

    if 'wrapper' in energy_model.named_steps:
        energy_model.named_steps['wrapper'].train()


def plot_loss_curves(losses: typing.Dict[str, typing.List[float]],
    title: str = 'Loss curve', xlabel: str = 'Epoch', ylabel: str = 'Loss',
    title_note: typing.Optional[str] = None, cmap: str = 'Dark2',
    legend_loc: str = 'best'):

    cm = plt.get_cmap(cmap)  # type: ignore

    for i, (key, loss_list) in enumerate(losses.items()):
        plt.plot(loss_list, label=key, color=cm(i))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title_note is not None:
        plt.title(f'{title} ({title_note})')
    else:
        plt.title(title)

    plt.legend(loc=legend_loc)
    plt.show()


def print_results_dict(results: typing.Dict[str, typing.Dict[str, typing.Any]],
                       results_keys: typing.Optional[typing.List[str]] = None, notebook: bool = True):
    if results_keys is None:
        results_keys = list(results.keys())

    for results_key in results_keys:
        results_dict = results[results_key]
        if results_dict is not None:
            if notebook:
                display(Markdown(f'### {results_key.capitalize()} results:'))
            else:
                display(f'{results_key.capitalize()} results:')
            display(results_dict)


CV_RESULTS_KEY_PATTERNS = ['mean_test_{name}', 'std_test_{name}', 'rank_test_{name}']
CV_RESULTS_COLUMN_PATTERNS = ['{name}_mean', '{name}_std', '{name}_rank']


def visualize_cv_outputs(cv: GridSearchCV, train_tensor: torch.Tensor,
    test_tensor: typing.Optional[torch.Tensor] = None,
    results: typing.Optional[typing.Dict[str, typing.Dict[str, float]]] = None,
    display_metric_correlation_table: bool = True,
    display_results_by_metric: bool = True,
    display_energy_histogram: bool = True, histogram_title_base: str = 'Energy scores of all games',
    title_note: typing.Optional[str] = None, histogram_log_y: bool = True,
    dispaly_weights_histogram: bool = True, weights_histogram_title_base: str = 'Energy model weights',
    cv_key_patterns: typing.List[str] = CV_RESULTS_KEY_PATTERNS,
    cv_column_patterns: typing.List[str] = CV_RESULTS_COLUMN_PATTERNS,
    cv_column_prefix: str = 'mean_test_', notebook: bool = True,
    ) -> None:

    if len(cv_key_patterns) != len(cv_column_patterns):
        raise ValueError(f'cv_key_patterns and cv_column_patterns must have the same length, but got {len(cv_key_patterns)} and {len(cv_column_patterns)} respectively.')

    scoring_names = cv.scorer_.names if hasattr(cv.scorer_, 'names') else []  # type: ignore

    if not scoring_names:
        scoring_names = [column.replace(cv_column_prefix, '') for column in cv.cv_results_.keys() if column.startswith(cv_column_prefix)]

    cv_series_or_dfs = [
        pd.Series(cv.cv_results_[key_pattern.format(name=name)], name=column_pattern.format(name=name))
        for (key_pattern, column_pattern) in zip(cv_key_patterns, cv_column_patterns)
        for name in scoring_names
    ]
    cv_series_or_dfs.insert(0, pd.DataFrame(cv.cv_results_["params"]))  # type: ignore

    # cv_series_or_dfs = [
    #     pd.DataFrame(cv.cv_results_["params"]),
    #     pd.Series(cv.cv_results_["mean_test_overall_ecdf"], name='ecdf_mean'),
    #     pd.Series(cv.cv_results_["std_test_overall_ecdf"], name='ecdf_std'),
    #     pd.Series(cv.cv_results_["rank_test_overall_ecdf"], name='ecdf_rank'),
    #     pd.Series(cv.cv_results_["mean_test_single_game_rank"], name='game_rank_mean'),
    #     pd.Series(cv.cv_results_["std_test_single_game_rank"], name='game_rank_std'),
    #     pd.Series(cv.cv_results_["rank_test_single_game_rank"], name='game_rank_rank'),
    # ]

    cv_df = pd.concat(cv_series_or_dfs, axis=1)

    if results is not None:
        print_results_dict(results, notebook=notebook)

    if display_metric_correlation_table:
        n_score_funcs = len(scoring_names)
        if notebook:
            display(Markdown('### Rank (Spearman) correlations between metrics:'))
        else:
            logger.debug('Rank (Spearman) correlations between metrics:')

        metrics_table = [[''] * n_score_funcs for _ in range(n_score_funcs)]

        for first_index, second_index in combinations(range(n_score_funcs), 2):
            first_name = scoring_names[first_index]
            second_name = scoring_names[second_index]
            first_rank = cv_df[f'{first_name}_rank']
            second_rank = cv_df[f'{second_name}_rank']
            corr = spearmanr(first_rank, second_rank)
            corr_str = f'{corr.statistic:.3f} (p={corr.pvalue:.4f})'  # type: ignore
            metrics_table[first_index][second_index] = corr_str
            metrics_table[second_index][first_index] = corr_str

        for index, name in enumerate(scoring_names):
            metrics_table[index].insert(0, f'**{name}**')

        table = tabulate(metrics_table, headers=[''] + scoring_names, tablefmt='github')
        if notebook:
            display(Markdown(table))
        else:
            logger.debug(f'Metric correlation table:\n{table}')

    if display_results_by_metric:
        for name in scoring_names:
            if notebook:
                display(Markdown(f'### CV results by {name}:'))
                display(cv_df.sort_values(by=f'{name}_rank').head(10))
            else:
                logger.debug(f'CV results by {name}:\n{cv_df.sort_values(by=f"{name}_rank").head(10)}')

    # else:
    #     if display_by_ecdf:
    #         display(Markdown('### CV results by overall ECDF:'))
    #         display(cv_df.sort_values(by='ecdf_rank').head(10))

    #     if display_by_game_rank:
    #         display(Markdown('### CV results by mean single game rank:'))
    #         display(cv_df.sort_values(by='game_rank_rank').head(10))

    if notebook and display_energy_histogram:
        plot_energy_histogram(cv, train_tensor, test_tensor, histogram_title_base, title_note, histogram_log_y)

    fitness_model = typing.cast(SklearnFitnessWrapper, cv.best_estimator_.named_steps['fitness'])  # type: ignore

    if notebook and fitness_model.losses:
        plot_loss_curves(fitness_model.losses, 'Fitness model loss curve', title_note=title_note)

    if notebook and dispaly_weights_histogram:
        fc1 = typing.cast(torch.nn.Linear, fitness_model.model.fc1)
        weights = fc1.weight.data.detach().numpy().squeeze()
        bias = fc1.bias.data.detach().numpy().squeeze()
        print(f'Weights mean: {weights.mean():.3f} +/- {weights.std():.3f} with bias {bias:.3f}')

        bins = max(min(50, len(weights) // 10), 10)
        plt.hist(weights, bins=bins)

        if title_note is not None:
            plt.title(f'{weights_histogram_title_base} ({title_note})')
        else:
            plt.title(weights_histogram_title_base)

        plt.xlabel('Weight magnitude')
        plt.ylabel('Count')
        plt.show()



HTML_DIFF = HtmlDiff(wrapcolumn=80)
HTML_DIFF_SUBSTITUTIONS = {
    'td.diff_header {text-align:right}': '.diff td {text-align: left !important}\n.diff th {text-align: center!important }\n.diff td.diff_header {text-align:right !important}',
    '.diff_add {background-color:#aaffaa}': '.diff_add {background-color: #6fa66f !important; font-weight: bold !important}',
    '.diff_chg {background-color:#ffff77}': '.diff_chg {background-color: #999949 !important; font-weight: bold !important}',
    '.diff_sub {background-color:#ffaaaa}': '.diff_sub {background-color: #a66f6f !important; font-weight: bold !important}',
    # '.diff_add {background-color:#aaffaa}': '.diff_add {color: #6fa66f; background-color: inherit; font-weight: bold}',
    # '.diff_chg {background-color:#ffff77}': '.diff_chg {color: #999949; background-color: inherit; font-weight: bold}',
    # '.diff_sub {background-color:#ffaaaa}': '.diff_sub {color: #a66f6f; background-color: inherit; font-weight: bold}',
}


def display_game_diff_html(before: str, after: str, html_diff_substitutions: typing.Dict[str, str] = HTML_DIFF_SUBSTITUTIONS):
    diff = HTML_DIFF.make_file(before.splitlines(), after.splitlines())  #, context=True, numlines=0)

    for key, value in html_diff_substitutions.items():
        diff = diff.replace(key, value)

    display(HTML(diff))


def evaluate_single_game_energy_contributions(cv: typing.Union[GridSearchCV, Pipeline], game_features: torch.Tensor, game_text: str,
    feature_names: typing.List[str], top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005,):

    energy_model = cv
    if isinstance(cv, GridSearchCV):
        energy_model = cv.best_estimator_
    weights = energy_model['fitness'].model.fc1.weight.data.detach().squeeze()  # type: ignore

    index_energy_contributions = game_features * weights

    real_game_energy = energy_model.transform(game_features).item()  # type: ignore

    display(Markdown(f'### Energy of visualized game: {real_game_energy:.3f}'))

    if display_overall_features:
        display_energy_contributions_table(index_energy_contributions, game_features, weights, feature_names, top_k, min_display_threshold)

    if display_game:
        display(Markdown(f'### Game:'))
        display(Markdown(f'```pddl\n{game_text}\n```'))


def evaluate_energy_contributions(energy_model: typing.Union[GridSearchCV, Pipeline], data_tensor: torch.Tensor, index: typing.Union[int, typing.Tuple[int, int]],
    feature_names: typing.List[str], full_dataset_tensor: torch.Tensor,
    original_game_texts: typing.List[str], negative_game_texts: typing.List[str],
    index_in_negatives: bool = True, top_k: int = 10, display_overall_features: bool = False, display_relative_features: bool = True,
    display_features_pre_post_scaling: bool = False,
    display_game_diff: bool = True, html_diff_substitutions: typing.Dict[str, str] = HTML_DIFF_SUBSTITUTIONS, min_display_threshold: float = 0.0005,
    display_features_diff: bool = True) -> None:

    negatives = data_tensor[:, 1:, :]
    if isinstance(index, tuple):
        row, col = index
    else:
        if index_in_negatives:
            row, col = torch.div(index, negatives.shape[1], rounding_mode='trunc'), index % negatives.shape[1]
        else:
            row, col = torch.div(index, full_dataset_tensor.shape[1], rounding_mode='trunc') , index % full_dataset_tensor.shape[1]

    if index_in_negatives:
        index_features = negatives[row, col]
    else:
        index_features = full_dataset_tensor[row, col]

    real_game_features = data_tensor[row, 0]

    if index_in_negatives:
        original_game_index = (full_dataset_tensor[:, :2, :] == data_tensor[row, :2, :]).all(dim=-1).all(dim=-1).nonzero().item()
        print(f'Original game index: {original_game_index} | Negative game row: {row} | Negative game col: {col}')
    else:
        original_game_index = index

    original_game_text = original_game_texts[original_game_index]  # type: ignore
    negative_game_text = negative_game_texts[(original_game_index * negatives.shape[1]) + col]  # type: ignore

    evaluate_comparison_energy_contributions(
        real_game_features, index_features, original_game_text,
        negative_game_text, energy_model, feature_names, top_k,
        display_overall_features, display_relative_features, display_game_diff,
        html_diff_substitutions, min_display_threshold, display_features_diff
    )

def evaluate_comparison_energy_contributions(
    original_game_features: torch.Tensor, comparison_game_features: torch.Tensor,
    original_game_text: str, comparison_game_text: str,
    energy_model: typing.Union[GridSearchCV, Pipeline],
    feature_names: typing.List[str], top_k: int = 10, display_overall_features: bool = False, display_relative_features: bool = True,
    display_game_diff: bool = True, html_diff_substitutions: typing.Dict[str, str] = HTML_DIFF_SUBSTITUTIONS, min_display_threshold: float = 0.0005,
    display_features_diff: bool = True
    ):

    if isinstance(energy_model, GridSearchCV):
        energy_model = energy_model.best_estimator_  # type: ignore

    energy_model = typing.cast(Pipeline, energy_model)

    if 'wrapper' in energy_model.named_steps:
        energy_model['wrapper'].eval()   # type: ignore

    index_energy = energy_model.transform(comparison_game_features).item()  # type: ignore
    real_game_energy = energy_model.transform(original_game_features).item()  # type: ignore

    weights = energy_model['fitness'].model.fc1.weight.data.detach().squeeze()  # type: ignore

    scaled_index_features = energy_model['scaler'].transform(comparison_game_features) if 'scaler' in energy_model else comparison_game_features   # type: ignore
    scaled_real_game_features = energy_model['scaler'].transform(original_game_features) if 'scaler' in energy_model else original_game_features  # type: ignore

    index_energy_contributions = scaled_index_features * weights
    real_game_contributions = scaled_real_game_features * weights

    display(Markdown(f'### Energy of real game: {real_game_energy:.3f} | Energy of regrown game: {index_energy:.3f} | Difference: {index_energy - real_game_energy:.3f}'))

    if display_overall_features:
        display_energy_contributions_table(index_energy_contributions, comparison_game_features, weights, feature_names, top_k, min_display_threshold)

    if display_relative_features:
        relative_contributions = index_energy_contributions - real_game_contributions
        display_energy_contributions_table(relative_contributions, comparison_game_features, weights, feature_names, top_k, min_display_threshold, original_game_features)

    if display_game_diff:
        display(Markdown('### Game Diffs'))
        display_game_diff_html(original_game_text, comparison_game_text, html_diff_substitutions)

    if display_features_diff:
        display(Markdown('### Feature Diffs'))
        d = comparison_game_features - original_game_features
        inds = d.nonzero().squeeze()
        if inds.ndim == 0 or len(inds) == 0:
            print('No features changed')

        else:
            diffs = d[inds]
            for i in torch.argsort(diffs):
                original_idx = inds[i]
                print(f'{feature_names[original_idx]}: {diffs[i]:.3f} ({scaled_real_game_features[original_idx]:.3f} => {scaled_index_features[original_idx]:.3f})')


def display_energy_contributions_table(energy_contributions: torch.Tensor, feature_values: torch.Tensor, weights: torch.Tensor,
        feature_names: typing.List[str], top_k: int, min_display_threshold: float = 0.005, real_game_features: typing.Optional[torch.Tensor] = None):
    energy_up_features = []
    energy_down_features = []

    top_k_contributions = torch.topk(energy_contributions, top_k, largest=True)
    if torch.any(top_k_contributions.values > min_display_threshold):
        for i in range(top_k):
            idx = top_k_contributions.indices[i]
            value = top_k_contributions.values[i]
            if value > min_display_threshold:
                # if display_features_pre_post_scaling:
                #     energy_up_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({real_game_features[idx]:.3f} => {scaled_real_game_features[idx]:.3f} | {index_features[idx]:.3f} => {scaled_index_features[idx]:.3f}) * {weights[idx]:.3f}')
                # else:
                if real_game_features is not None:
                    energy_up_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({real_game_features[idx]:.3f} => {feature_values[idx]:.3f}) * {weights[idx]:.3f}')
                else:
                    energy_up_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({feature_values[idx]:.3f}) * {weights[idx]:.3f}')

    bottom_k_contributions = torch.topk(energy_contributions, top_k, largest=False)
    if torch.any(bottom_k_contributions.values < -min_display_threshold):
        for i in range(top_k):
            idx = bottom_k_contributions.indices[i]
            value = bottom_k_contributions.values[i]
            if value < -min_display_threshold:
                # if display_features_pre_post_scaling:
                #     energy_down_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({real_game_features[idx]:.3f} => {scaled_real_game_features[idx]:.3f} | {index_features[idx]:.3f} => {scaled_index_features[idx]:.3f}) * {weights[idx]:.3f}')
                # else:
                if real_game_features is not None:
                    energy_down_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({real_game_features[idx]:.3f} => {feature_values[idx]:.3f}) * {weights[idx]:.3f}')
                else:
                    energy_down_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({feature_values[idx]:.3f}) * {weights[idx]:.3f}')

    if real_game_features is not None:
        display(Markdown(f'### Top features changing the game\'s energy\nfeature name: **value** = (original feature value => regrown feature value) * weight'))
    else:
        display(Markdown(f'### Top features contributing to the game\'s energy\nfeature name: **value** = (original feature value => regrown feature value) * weight'))

    rows = list(zip_longest(energy_up_features, energy_down_features))
    headers = ['Features increasing energy (= more fake)', 'Features decreasing energy (= more real)']
    table = tabulate(rows, headers=headers, tablefmt='github')
    display(Markdown(table))
