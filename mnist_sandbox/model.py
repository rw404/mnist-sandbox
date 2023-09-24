from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam


class MNISTNet(pl.LightningModule):
    # REQUIRED
    def __init__(self, learning_rate: float = 1e-3) -> None:
        super().__init__()
        """ Define computations here. """

        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.lr = learning_rate
        self.conv1 = nn.Conv2d(1, 32, 3, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 3, padding="same")
        self.conv3 = nn.Conv2d(64, 96, 3, padding="same")
        self.fc1 = nn.Linear(864, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    # REQUIRED
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use for inference only (separate from training_step)."""

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)

        return x

    # REQUIRED
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        """the full training loop"""
        x, y = batch

        y_logit = self(x)
        loss = F.cross_entropy(y_logit, y)

        classes = y_logit.data.max(1)[1]
        incorrect = classes.ne(y.long().data).cpu().sum()
        err = incorrect.item() / y.numel()
        acc = torch.tensor(1.0 - err)

        output = {"loss": loss, "acc": acc}

        self.training_step_outputs.append(output)

        return output

    # REQUIRED
    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Define optimizers and LR schedulers."""
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=1, verbose=True
        )
        lr_dict = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_acc",
        }

        return [optimizer], [lr_dict]

    # OPTIONAL
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        """the full validation loop"""
        x, y = batch

        y_logit = self(x)

        classes = y_logit.data.max(1)[1]
        incorrect = classes.ne(y.long().data).cpu().sum()
        err = incorrect.item() / y.numel()
        val_acc = torch.tensor(1.0 - err)

        loss = F.cross_entropy(y_logit, y)
        output = {"val_loss": loss, "val_acc": val_acc}

        self.validation_step_outputs.append(output)

        return output

    # OPTIONAL
    def on_train_epoch_end(self) -> None:
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs])
        avg_loss = avg_loss.mean()

        avg_acc = torch.stack([x["acc"] for x in self.training_step_outputs])
        avg_acc = avg_acc.mean()

        Accuracy = 100 * avg_acc.item()

        self.training_step_outputs.clear()

        print(f"| Train_loss: {avg_loss:.5f} Train_acc: {Accuracy}%")

        self.log("train_loss", avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", avg_acc, prog_bar=True, on_epoch=True, on_step=False)

    # OPTIONAL
    def on_validation_epoch_end(self) -> None:
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_acc = torch.stack([x["val_acc"] for x in self.validation_step_outputs]).mean()
        Accuracy = 100 * avg_acc.item()

        self.validation_step_outputs.clear()

        print(
            f"[Epoch {self.trainer.current_epoch:3}] "
            + f"Val_loss: {avg_loss:.5f} "
            + f"Val_accuracy: {Accuracy}%",
            end=" ",
        )

        self.log("val_loss", avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", avg_acc, prog_bar=True, on_epoch=True, on_step=False)
