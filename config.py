from dataclasses import dataclass


@dataclass
class OnnxPath:
    save_path: str


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
    logging_url: str


@dataclass
class TrainParams:
    data: Dataset
    model: Model
    trainer: Trainer


@dataclass
class DVCSetting:
    pth_endpoint: str


@dataclass
class InferParams:
    data: Dataset
    dvc_settings: DVCSetting


@dataclass
class InferenceSettings:
    pth_endpoint: str
    onnx_endpoint: str
    inference_input: str
    inference_output: str


@dataclass
class Inference:
    inference: InferenceSettings


@dataclass
class Converter:
    onnx_path: OnnxPath
    dvc_settings: DVCSetting
