import os
import shutil
from pathlib import Path

import hydra
import pandas as pd
import wandb
from conf import settings  # noqa
from loguru import logger
from omegaconf import DictConfig

from src.logger import set_logger
from src.create_dataset import get_dataset
from src.models.lgbm import train_and_predict
# from src.utils.utils_torch import seed_everything

ROOT_DIR = Path(__file__).parent.resolve()  # Path to current directory
INP_DIR = ROOT_DIR / "input" / "used_car_price_prediction"
NUM_WORKERS = min(os.cpu_count(), 10)


def train(log_dir: str, fold: int, hparams: dict) -> tuple[str, float]:
    # seed_everything(hparams.model.seed, deterministic=True, benchmark=False)
    # logger.info(f"Set seed with {hparams.model.seed}")

    # prepare dataset
    X_train, y_train, X_valid, y_valid, X_test = get_dataset(features=hparams["features"], obj=hparams["obj"], fold=fold)
    
    wandb.init(
        project="used_car_price_prediction",
        name=f"{hparams['name']}_fold{fold}",
        dir=log_dir,
        mode="online",
        tags=[hparams["model"]["name"], f"fold{fold}"],
    )

    # train lgbm
    pred_train, pred_valid, pred_test, model = train_and_predict(
        log_dir=log_dir,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        hparams=hparams,
    )

    # Save result
    pred = pd.read_csv(INP_DIR / "train_fold.csv")[["id", "fold"]].copy()
    pred.loc[pred["fold"] != fold, "pred"] = pred_train
    pred.loc[pred["fold"] == fold, "pred"] = pred_valid

    sub = pd.read_csv(INP_DIR / "submit_sample.csv", header=None)
    sub.columns = ["id", "pred"]
    sub["pred"] = pred_test

    pred.to_csv(log_dir / "pred.csv", index=False)
    sub.to_csv(log_dir / "sub.csv", index=False, header=False)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log_dir = ROOT_DIR / "logs" / cfg.exp.name / f"fold{cfg.fold}"
    set_logger(dir=log_dir)

    logger.info(f"{NUM_WORKERS=}")
    train(log_dir=log_dir, fold=cfg.fold, hparams=dict(cfg.exp))


if __name__ == "__main__":
    main()
