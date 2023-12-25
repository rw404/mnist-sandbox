import hydra
import torch
from config import TrainParams
from hydra.core.config_store import ConfigStore
from mnist_sandbox.data import MNIST
from mnist_sandbox.model import MNISTNet
from mnist_sandbox.train import train_m


cs = ConfigStore.instance()
cs.store(name="params", node=TrainParams)


@hydra.main(config_path="./configs", config_name="train", version_base="1.3")
def train_test(config: TrainParams) -> None:
    """
    Training process

    """
    device = torch.device("cpu")
    print(f"Device {device}")

    model = MNISTNet(learning_rate=config.model.learning_rate)
    model.to(device)

    print("Data init...")
    dataset = MNIST(
        path_list=config.data.dataset_list,
        seed=config.data.seed,
        batch_size_train=config.data.batch_size,
        train=True,
    )

    print("Training...")
    model = train_m(
        model,
        config.trainer.epochs,
        dataset.train_loader,
        dataset.val_loader,
        logging_url=config.trainer.logging_url,
    )


if __name__ == "__main__":
    train_test()
