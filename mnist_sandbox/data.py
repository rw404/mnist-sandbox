import torchvision
from torch.utils.data import DataLoader, random_split


batch_size_train = 64
batch_size_test = 10

train_dataset = torchvision.datasets.MNIST(
    "./",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

train_len = int(len(train_dataset) * 0.8)
val_len = len(train_dataset) - int(len(train_dataset) * 0.8)
train_set, val_set = random_split(
    train_dataset,
    [train_len, val_len],
)

train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)

val_loader = DataLoader(val_set, batch_size=batch_size_train, shuffle=True)

test_loader = DataLoader(
    torchvision.datasets.MNIST(
        "./",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_test,
    shuffle=False,
)
