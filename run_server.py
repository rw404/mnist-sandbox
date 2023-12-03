import io
import tempfile
from pathlib import Path

import dvc.api
import hydra
import mlflow.onnx
import numpy as np
import onnx
import torch
from config import Inference
from hydra.core.config_store import ConfigStore
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from mnist_sandbox.model import MNISTNet


cs = ConfigStore.instance()
cs.store(name="params", node=Inference)


@hydra.main(config_path="./configs", config_name="inference", version_base="1.3")
def runtime(config: Inference):
    model_info = dvc.api.read(
        str(Path("models") / config.inference.pth_endpoint),
        repo="https://github.com/rw404/mnist-sandbox",
        mode="rb",
    )

    device = torch.device("cpu")
    print(f"Device {device}")

    model = MNISTNet()
    model.to(device)

    print("Loading model...")
    model.load_state_dict(torch.load(io.BytesIO(model_info)))

    with tempfile.TemporaryDirectory(dir="./") as tmpp:
        model_info_x = dvc.api.read(
            config.inference.onnx_endpoint,
            repo="https://github.com/rw404/mnist-sandbox",
            mode="rb",
        )
        with open(Path(tmpp) / config.inference.onnx_endpoint, "wb+") as buffer:
            buffer.write(model_info_x)

        onnx_model = onnx.load_model(Path(tmpp) / config.inference.onnx_endpoint)

        inpp = torch.randn((1, 1, 28, 28))

        with mlflow.start_run():
            signature = infer_signature(inpp.numpy(), model(inpp).detach().numpy())
            model_info = mlflow.onnx.log_model(onnx_model, "model", signature=signature)

        model_from_mlflow = mlflow.pyfunc.load_model(model_info.model_uri)

    test_image = np.load(Path("inference") / config.inference.inference_input)[None, ...]

    predictions = model_from_mlflow.predict(test_image)["26"]

    digit = predictions.argmax()

    plt.figure(figsize=(6, 6))
    plt.title(f"Prediction is {digit}", fontsize=15)
    plt.imshow(test_image[0][0], cmap="gray")
    plt.savefig(Path("inference") / config.inference.inference_output)


if __name__ == "__main__":
    runtime()
