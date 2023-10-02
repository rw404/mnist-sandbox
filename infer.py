import os

import numpy as np
import pandas as pd
import torch
from mnist_sandbox.data import test_loader
from mnist_sandbox.eval import model_evaluate
from mnist_sandbox.model import MNISTNet


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")

    model = MNISTNet()
    model.to(device)

    print("Loading model...")
    model.load_state_dict(
        torch.load(os.path.join(os.path.abspath("./"), "sota_mnist_cnn.pth"))
    )

    print("Evaluating...")
    model.eval()
    correct, total, prediction_list = model_evaluate(model, test_loader)

    print("Saving submission...")
    pred_df = {"ImageId": np.arange(prediction_list.shape[0]), "Label": prediction_list}

    result_submission = pd.DataFrame(pred_df)
    result_submission.to_csv("submission.csv", index=False)
    print(f"Done. Accuracy[test dataset] = {(100 * correct / total):.3f}%")
