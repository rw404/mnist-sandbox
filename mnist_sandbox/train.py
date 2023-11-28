from pathlib import Path

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
    Returns
    -------
    model: pl.LightningModule
        Trained model.
    """

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=n_epochs,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    if not Path("models").exists():
        Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path("models") / "sota_mnist_cnn.pth")
    weights_path = Path.cwd().joinpath() / "models" / "sota_mnist_cnn.pth"
    print(f"Model saved into {weights_path}")
    return model
