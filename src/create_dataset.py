from pathlib import Path

import pandas as pd
from loguru import logger

ROOT_DIR = Path(__file__).parent.resolve().parent
INP_DIR = ROOT_DIR / "input" / "used_car_price_prediction"
FEAT_DIR = ROOT_DIR / "features"


def get_dataset(
    features: list[str], obj: str, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        features (list[str]): _description_
        fold (int): _description_

    Returns:
        X_train, y_train, X_valid, y_valid, X_test
    """
    train = pd.read_csv(INP_DIR / "train_fold.csv").query("fold != @fold")[["id"]]
    valid = pd.read_csv(INP_DIR / "train_fold.csv").query("fold == @fold")[["id"]]
    test = pd.read_csv(INP_DIR / "test.csv")[["id"]]

    for feat in features:
        tmp_train = pd.read_csv((FEAT_DIR / "train" / feat).with_suffix(".csv")).query("fold != @fold")
        tmp_test = pd.read_csv((FEAT_DIR / "test" / feat).with_suffix(".csv"))

        train = train.merge(tmp_train[["id", feat]], how="left", on="id")
        valid = valid.merge(tmp_train[["id", feat]], how="left", on="id")
        test = test.merge(tmp_test, on="id")

    X_train = train[features].reset_index(drop=True)
    X_valid = valid[features].reset_index(drop=True)
    X_test = test[features].reset_index(drop=True)
    y_train = pd.read_csv((FEAT_DIR / "train" / obj).with_suffix(".csv")).query("fold != @fold")[[obj]].reset_index(drop=True)
    y_valid = pd.read_csv((FEAT_DIR / "train" / obj).with_suffix(".csv")).query("fold == @fold")[[obj]].reset_index(drop=True)

    logger.info(f"{X_train.shape=}, {X_valid.shape=}, {X_test.shape=}, {y_train.shape=}, {y_valid.shape=}")
    return X_train, y_train, X_valid, y_valid, X_test
