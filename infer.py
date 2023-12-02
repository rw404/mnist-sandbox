import io
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
import torch
from mnist_sandbox import RANDOM_SEED
from mnist_sandbox.data import MNIST
from mnist_sandbox.eval import model_evaluate
from mnist_sandbox.model import MNISTNet


def inference() -> None:
    """
    Inference model with csv saving
    """
    model_info = dvc.api.read(
        str(Path("models") / "sota_mnist_cnn.pth"),
        repo="https://github.com/rw404/mnist-sandbox",
        mode="rb",
    )

    device = torch.device("cpu")
    print(f"Device {device}")

    model = MNISTNet()
    model.to(device)

    print("Loading model...")
    model.load_state_dict(torch.load(io.BytesIO(model_info)))

    print("Data init...")
    dataset = MNIST(seed=RANDOM_SEED)

    print("Evaluating...")
    model.eval()
    correct, total, prediction_list = model_evaluate(model, dataset.test_loader)

    print("Saving submission...")
    pred_df = {"ImageId": np.arange(prediction_list.shape[0]), "Label": prediction_list}

    result_submission = pd.DataFrame(pred_df)
    result_submission.to_csv("submission.csv", index=False)
    print(f"Done. Accuracy[test dataset] = {(100 * correct / total):.3f}%")


if __name__ == "__main__":
    inference()