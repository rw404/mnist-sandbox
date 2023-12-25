# mnist-sandbox

![pre-commit workflow](https://github.com/rw404/MNIST_sandbox/actions/workflows/pre-commit.yml/badge.svg)
<a href="https://rw404.github.io/MNIST_sandbox/">
<img src="https://img.shields.io/badge/GitHub%20Pages-222222?style=for-the-badge&logo=GitHub%20Pages&logoColor=white" />
</a>

Задача классификации рукописных цифр с соревнования
[kaggle](https://www.kaggle.com/competitions/digit-recognizer).

> Учебный проект для курса MLops 2023

## Install dependencies

```bash
poetry install [OPTIONAL]--with docs
```

## Build

```bash
poetry build
```

## Tests HW

```bash
mkdir mnist_test
virtualenv mnist_test
source ./mnist_test/bin/activate

poetry install

pre-commit install

pre-commit run -a

python3 train.py
python3 infer.py
```

# 2nd HW test

> In the 2nd HW it is used dvc for downloading, so each time in traininig it
> downloads required dataset in tmp folder which will be deleted after moving
> data to RAM, so for model loading in eval/inference modes

```bash
# Logging
# Step 1: enable docker image of local mlflow.
# Using ports 13412, 13413. Have to be free
cd logger_mlflow
docker-compose build
docker-compose up

# Step 2: run training script
# Return to the repository directory
cd ..
# Check training yaml
cat configs/train.yaml
# Run training
python3 train.py

# Step 3: check metrics
open localhost:13412

# Inference
# Step 1: specify inference setting in configs/inference.yaml
# Default is:
# inference:
#   pth_endpoint: sota_mnist_cnn.pth <- last model | from dvc
#   onnx_endpoint: mnist_cnn.onnx <- last onnx     | from dvc
#   inference_input: inference.npy <- image | required numpy (1, 28, 28)
#   inference_output: inference_predict.png <- where to store the prediction

# Step 2: run inference
# Because of lots of troubleshooting with automatic
# model serving with mlflow it's not used request-based approach
# But mlflow was used as saver and loader onnx weights
python3 run_server.py

# Step 3: check result with title of prediction
open inference/inference_predict.png
```

# 3rd HW

Информация в файле triton.md
