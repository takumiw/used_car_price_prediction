import pickle
from pathlib import PosixPath

import lightgbm as lgb
import numpy as np
# import wandb
import pandas as pd
from lightgbm import Booster, early_stopping
from loguru import logger
from wandb.lightgbm import wandb_callback, log_summary

from src.logger import log_evaluation




def train_and_predict(
    log_dir: PosixPath,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    hparams: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Booster]:
    """Train LightGBM
    Args:
    Returns:
        pred_train (np.ndarray): raw prediction of train
        pred_valid (np.ndarray): raw prediction of validation
        pred_test (np.ndarray): raw prediction of test
        model (Booster): trained model
    """
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    callbacks = [
        early_stopping(stopping_rounds=hparams["params"]["early_stopping_round"]),
        log_evaluation(period=hparams["params"]["verbose_round"]),
        wandb_callback(),
    ]

    model = lgb.train(
        dict(hparams["hparams"]),
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        feval=eval_accuracy if hparams["hparams"]["metric"] == "accuracy" else None,
        num_boost_round=hparams["params"]["num_iterations"],
        callbacks=callbacks,
        categorical_feature=hparams["cat_features"],
    )

    with open(log_dir / "model.pkl", mode="wb") as f:
        pickle.dump(model, f)

    log_summary(model, save_model_checkpoint=False)

    logger.info(f"best iteration: {model.best_iteration}")
    logger.info(f'training best score: {model.best_score["training"].items()}')
    logger.info(f'valid_1 best score: {model.best_score["valid_1"].items()}')

    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    return pred_train, pred_valid, pred_test, model


def eval_accuracy(preds: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
    n_classes = 15
    y_true = data.get_label()
    reshaped_preds = preds.reshape(n_classes, len(preds) // n_classes)
    y_pred = np.argmax(reshaped_preds, axis=0)
    acc = np.mean(y_true == y_pred)
    return "accuracy", acc, True
