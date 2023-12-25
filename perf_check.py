import io
from pathlib import Path

import dvc.api
import hydra
import torch
from config import Converter
from hydra.core.config_store import ConfigStore
from mnist_sandbox.model import MNISTNet
from thop import clever_format, profile


cs = ConfigStore.instance()
cs.store(name="params", node=Converter)

BATCH_SIZE = 13


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
    model.eval()

    print("Perf measure...")

    input = torch.randn(BATCH_SIZE, 1, 28, 28)
    output = torch.randn(BATCH_SIZE, 10)
    old_macs, old_params = profile(model, inputs=(input,))

    old_params += input.numel()
    old_params += output.numel()

    macs, params = clever_format([old_macs, 4 * old_params], "%.3f")
    print(f"OPS: {macs}, BYTES: {params} #ops/#bytes = {old_macs/(4*old_params)}")


if __name__ == "__main__":
    convert()
