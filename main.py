from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from data import train_loader, val_loader, test_loader
from model import MNISTNet
from train import train_m
from eval import model_evaluate
import warnings

warnings.filterwarnings('ignore')

n_epochs = 3
learning_rate = 1e-3

random_seed = 1
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")

    model = MNISTNet(learning_rate=learning_rate)
    model.to(device)

    print("Training...")
    model = train_m(model, n_epochs, train_loader, val_loader)

    print("Saving...")
    torch.save(model.state_dict(), "sota_mnist_cnn.pth")

    print("Validation...")
    model.eval()
    correct, total = model_evaluate(model, test_loader)

    print(f"Accuracy = {100*correct/total}%")
