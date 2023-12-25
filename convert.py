import io
from pathlib import Path

import dvc.api
import hydra
import torch
from config import Converter
from hydra.core.config_store import ConfigStore
from mnist_sandbox.model import MNISTNet


cs = ConfigStore.instance()
cs.store(name="params", node=Converter)


@hydra.main(config_path="./configs", config_name="convert", version_base="1.3")
def convert(config: Converter) -> None:
    """
    Inference model with csv saving
    """
    model_info = dvc.api.read(
        str(Path("models") / config.dvc_settings.pth_endpoint),
        repo="https://github.com/rw404/mnist-sandbox",
        mode="rb",
    )

    device = torch.device("cpu")
    print(f"Device {device}")

    model = MNISTNet()
    model.to(device)

    print("Loading model...")
    model.load_state_dict(torch.load(io.BytesIO(model_info)))

    print("Converting model...")
    model.to_onnx(
        config.onnx_path.save_path, torch.randn((1, 1, 28, 28)), export_params=True
    )


if __name__ == "__main__":
    convert()
