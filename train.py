import torch
from mnist_sandbox.data import train_loader, val_loader
from mnist_sandbox.model import MNISTNet
from mnist_sandbox.train import train_m


n_epochs = 2
learning_rate = 1e-3

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")

    model = MNISTNet(learning_rate=learning_rate)
    model.to(device)

    print("Training...")
    model = train_m(model, n_epochs, train_loader, val_loader)
