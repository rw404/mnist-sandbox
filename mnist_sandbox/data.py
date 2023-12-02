import tempfile
from pathlib import Path

import dvc.api
import torch
import torchvision
from torch.utils.data import DataLoader, random_split


class MNIST:
    """MNIST Dataset

    * Contains MNIST data
    * Downloads & initialize dataloaders + saves data locally


    """

    def __init__(
        self,
        path_list: list[str],
        batch_size_train=64,
        batch_size_test=10,
        seed=42,
        train=True,
    ) -> None:
        # TODO: remove hardcoded
        if path_list is None:
            print("path_list parameter must be specified! Exiting")
            exit(0)

        with tempfile.TemporaryDirectory(dir="./") as tmpp:
            root_dir = Path(tmpp) / "MNIST"
            root_dir.mkdir(parents=True, exist_ok=True)

            root_dir = root_dir / "raw"
            root_dir.mkdir(parents=True, exist_ok=True)

            for image_file in path_list:
                cur_file = dvc.api.read(
                    str(Path("MNIST") / "raw" / image_file),
                    repo="https://github.com/rw404/mnist-sandbox",
                    mode="rb",
                )

                with open(root_dir / image_file, mode="wb+") as data_file:
                    data_file.write(cur_file)

            if train:
                train_dataset = torchvision.datasets.MNIST(
                    tmpp,
                    train=True,
                    download=False,
                    transform=torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                )

                self.train_len = int(len(train_dataset) * 0.8)

                self.val_len = len(train_dataset) - int(len(train_dataset) * 0.8)

                generator = torch.Generator().manual_seed(seed)
                self.train_set, self.val_set = random_split(
                    train_dataset, [self.train_len, self.val_len], generator=generator
                )

                self.train_loader_instance = DataLoader(
                    self.train_set, batch_size=batch_size_train, shuffle=True
                )

                self.val_loader_instance = DataLoader(
                    self.val_set, batch_size=batch_size_train, shuffle=True
                )
            else:
                self.test_loader_instance = DataLoader(
                    torchvision.datasets.MNIST(
                        tmpp,
                        train=False,
                        download=False,
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

    @property
    def train_loader(self):
        return self.train_loader_instance

    @property
    def val_loader(self):
        return self.val_loader_instance

    @property
    def test_loader(self):
        return self.test_loader_instance
