import torch

from . import LEARNING_RATE, N_EPOCHS, RANDOM_SEED
from .data import MNIST
from .eval import model_evaluate
from .model import MNISTNet
from .train import train_m


torch.manual_seed(RANDOM_SEED)


def main() -> None:
    """
    Run e2e pipeline for MNIST CNN classifier
    """

    device = torch.device("cpu")
    print(f"Device {device}")

    model = MNISTNet(learning_rate=LEARNING_RATE)
    model.to(device)

    print("Data init...")
    dataset = MNIST(seed=RANDOM_SEED)

    print("Training...")
    model = train_m(model, N_EPOCHS, dataset.train_loader, dataset.val_loader)

    print("Validation...")
    model.eval()
    correct, total, _ = model_evaluate(model, dataset.test_loader)

    print(f"Accuracy = {100*correct/total}%")


if __name__ == "__main__":
    main()
