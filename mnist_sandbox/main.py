import warnings

import torch
from data import test_loader, train_loader, val_loader
from eval import model_evaluate
from model import MNISTNet
from train import train_m


n_epochs = 2
learning_rate = 1e-3
random_seed = 1

warnings.filterwarnings("ignore")

torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")

    model = MNISTNet(learning_rate=learning_rate)
    model.to(device)

    print("Training...")
    model = train_m(model, n_epochs, train_loader, val_loader)

    print("Validation...")
    model.eval()
    correct, total, _ = model_evaluate(model, test_loader)

    print(f"Accuracy = {100*correct/total}%")
