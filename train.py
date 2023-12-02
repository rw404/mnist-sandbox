import torch
from mnist_sandbox import LEARNING_RATE, N_EPOCHS, RANDOM_SEED
from mnist_sandbox.data import MNIST
from mnist_sandbox.model import MNISTNet
from mnist_sandbox.train import train_m


def train_test() -> None:
    """
    Training process

    """
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
