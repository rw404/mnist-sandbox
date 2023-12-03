import subprocess
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader


def train_m(
    model: pl.LightningModule,
    n_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save: bool = False,
    logging_url: str = "file:./.logs/my-mlflow-logs",
) -> pl.LightningModule:
    """Training model on train_loader and validation part with val_loader.

    * Save trained model weights into sota_mnist_cnn.pth

    Parameters
    ----------
    model: pl.LightningModule
        Model to be trained.
    n_epochs: int
        Count of training epochs.
    train_loader: DataLoader
        Training dataset.
    val_loader: DataLoader
        Validation dataset.
    save: bool
        Save result model. Default is False, because DVC is used
    logging_url: str
        Where to store mlflow logs
    Returns
    -------
    model: pl.LightningModule
        Trained model.
    """

    # Git commit
    git_commit_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode("utf-8")
    )

    logger = MLFlowLogger(experiment_name=git_commit_id, tracking_uri=logging_url)

    logger.log_hyperparams(dict(model.hparams).update({"git_commit_id": git_commit_id}))

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=n_epochs,
        logger=logger,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    if save:
        if not Path("models").exists():
            Path("models").mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), Path("models") / "sota_mnist_cnn.pth")
        weights_path = Path.cwd().joinpath() / "models" / "sota_mnist_cnn.pth"
        print(f"Model saved into {weights_path}")
    else:
        print("Model stored in DVC, so it's not saving.")
    return model
