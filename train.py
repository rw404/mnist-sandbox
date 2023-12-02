import hydra
import torch
from config import Params
from hydra.core.config_store import ConfigStore
from mnist_sandbox import LEARNING_RATE, N_EPOCHS, RANDOM_SEED
from mnist_sandbox.data import MNIST
from mnist_sandbox.model import MNISTNet
from mnist_sandbox.train import train_m


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="./configs", config_name="train", version_base="1.3")
def train_test(cfg: Params) -> None:
    """
    Training process

    """
    print(cfg.data)
    print(cfg.data.path)
    print(cfg.data.dvc_shell_dwnld)

    device = torch.device("cpu")
    print(f"Device {device}")

    model = MNISTNet(learning_rate=LEARNING_RATE)
    model.to(device)

    print("Data init...")
    dataset = MNIST(seed=RANDOM_SEED)

    print("Training...")
    model = train_m(model, N_EPOCHS, dataset.train_loader, dataset.val_loader)


if __name__ == "__main__":
    train_test()
