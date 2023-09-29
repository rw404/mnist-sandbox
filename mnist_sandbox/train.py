import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


def train_m(
    model: pl.LightningModule,
    n_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> pl.LightningModule:
    """Training model on train_loader and validation part with val_loader.

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
    Returns
    -------
    model: pl.LightningModule
        Trained model.
    """

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=n_epochs,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    return model
