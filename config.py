from dataclasses import dataclass


@dataclass
class Dataset:
    dataset_list: str
    seed: int
    batch_size: int


@dataclass
class Model:
    learning_rate: float


@dataclass
class Trainer:
    epochs: int


@dataclass
class TrainParams:
    data: Dataset
    model: Model
    trainer: Trainer


@dataclass
class InferParams:
    data: Dataset
