import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


def train_m(
    model: pl.LightningModule,
    n_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> pl.LightningModule:
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=n_epochs,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    return model
