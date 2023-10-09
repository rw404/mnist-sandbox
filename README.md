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

## Run

```bash
poetry run python3 -m mnist_sandbox.main
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
