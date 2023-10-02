from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# @torch.no_grad() -- removed for sphinx docstring
def model_evaluate(
    model: pl.LightningModule, test_loader: DataLoader
) -> Tuple[int, int, np.ndarray]:
    """Evaluate model with specified test dataset.

    * Evaluation without torch.grad calculations.

    Parameters
    ----------
    model: pl.LightningModule
        Trained model in inference mode.
    test_loader: DataLoader
        Test dataset to be evaluated on.
    Returns
    -------
    correct: int
        Count of correct predictions.
    total: int
        Count of all items.
    """
    prediction = []

    with torch.no_grad():
        correct = 0
        total = 0
        for img, label in tqdm(test_loader):
            pred = model(img).detach().cpu()

            correct += (pred.argmax(dim=1) == label).sum()
            prediction.append(pred.argmax(dim=1).numpy())

            total += pred.size(0)

    prediction_list = np.hstack(prediction)

    return correct, total, prediction_list
