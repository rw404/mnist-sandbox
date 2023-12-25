# Описание применения сервинга модели с использованием [Triton Server](https://developer.nvidia.com/triton-inference-server)

- В
  [домашнем задании 2](https://github.com/rw404/mnist-sandbox/releases/tag/2ndHW)
  использовалась модель в формате ONNX, поэтому сервинг модели будет
  использовать в качестве бэкенда ONNX(CPU).

## Системная конфигурация

- OS: Windows 11 + WSL2
- CPU: 12th Gen Intel(R) Core(TM) i9-12900F
- vCPU: 24
- RAM: 32 GB
- Nvidia-Triton docker version: 23.04-py3. Запуск специализирован для CPU(в
  случае onnx), но такая версия подходит для GPU с cuda >= 12.0.

## Описание решаемой задачи

Задача классификации рукописных цифр с соревнования
[kaggle](https://www.kaggle.com/competitions/digit-recognizer) с использованием
базовой сверточной сети.

## Описание структуры model_repository

```bash
$ tree model_repository
model_repository/
└── onnx-mnist-cpu
    ├── 1
    └── config.pbtxt

2 directories, 1 file
```

### Сборка модели

```bash
poetry run python3 convert.py
cd triton_backend_cpu
docker-compose build
docker-compose up
```

Для вызова клиента необходимо вызвать `triton_backend_cpu/client.py`:

```bash
cd triton_backend_cpu
poetry run python3 client.py
# запускал python3 client.py,
# так как была ошибка с
# внутренней адресацией портов через poetry
```

Ожидаемый вывод(скачиваются с dvc изображения, проверяются результаты):

```bash
Test for image of 0 class is done.
Test for image of 8 class is done.
```

## Throughput & latency

До всех оптимизаций следующие метрики показателей эффективности:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 709.111 infer/sec, latency 1402 usec
Concurrency: 2, throughput: 1247.22 infer/sec, latency 1606 usec
Concurrency: 3, throughput: 1388.56 infer/sec, latency 2158 usec
Concurrency: 4, throughput: 1716.81 infer/sec, latency 2325 usec
Concurrency: 5, throughput: 1739.27 infer/sec, latency 2878 usec
Concurrency: 6, throughput: 2073.62 infer/sec, latency 2891 usec
Concurrency: 7, throughput: 1924.72 infer/sec, latency 3637 usec
Concurrency: 8, throughput: 2108.97 infer/sec, latency 3792 usec
```

После подбора оптимальных параметров(секция **Выбор оптимизаций**)

```yaml
instance_group [
    {
        count: 5
        kind: KIND_CPU
    }
]
dynamic_batching: { max_queue_delay_microseconds: 150 }
```

Следующие показатели(итог оптимизаций):

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 656.16 infer/sec, latency 1523 usec
Concurrency: 2, throughput: 1371.09 infer/sec, latency 1459 usec
Concurrency: 3, throughput: 1652.49 infer/sec, latency 1815 usec
Concurrency: 4, throughput: 1893.37 infer/sec, latency 2111 usec
Concurrency: 5, throughput: 2180.52 infer/sec, latency 2291 usec
Concurrency: 6, throughput: 2466.73 infer/sec, latency 2432 usec
Concurrency: 7, throughput: 2552.49 infer/sec, latency 2742 usec
Concurrency: 8, throughput: 2719.33 infer/sec, latency 2939 usec
```

## Выбор оптимизаций

В силу описания файла model.onnx изображение поступает на вход только размеров
[1, 1, 28, 28]. Поэтому параметр max_batch_size был исключен из рассмотрения.

Для тестирования создавался контейнер:

```bash
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.04-py3-sdk
```

Затем вводилась команда(тесты запускались такой командой):

```bash
perf_analyzer -m onnx-mnist-cpu -u localhost:8500 --concurrency-range 1:8 --shape input.1:1,1,28,28
```

Изначальный запуск выдал следующие показатели:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 709.111 infer/sec, latency 1402 usec
Concurrency: 2, throughput: 1247.22 infer/sec, latency 1606 usec
Concurrency: 3, throughput: 1388.56 infer/sec, latency 2158 usec
Concurrency: 4, throughput: 1716.81 infer/sec, latency 2325 usec
Concurrency: 5, throughput: 1739.27 infer/sec, latency 2878 usec
Concurrency: 6, throughput: 2073.62 infer/sec, latency 2891 usec
Concurrency: 7, throughput: 1924.72 infer/sec, latency 3637 usec
Concurrency: 8, throughput: 2108.97 infer/sec, latency 3792 usec
```

### vCPU

Попробуем оценить влияние числа vCPU. То есть изменим параметр `count: 24` на
`count: 12`:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 694.299 infer/sec, latency 1439 usec
Concurrency: 2, throughput: 1515 infer/sec, latency 1318 usec
Concurrency: 3, throughput: 1539.66 infer/sec, latency 1946 usec
Concurrency: 4, throughput: 1698.4 infer/sec, latency 2355 usec
Concurrency: 5, throughput: 2009.91 infer/sec, latency 2487 usec
Concurrency: 6, throughput: 2052.53 infer/sec, latency 2922 usec
Concurrency: 7, throughput: 1816.07 infer/sec, latency 3852 usec
Concurrency: 8, throughput: 2227.58 infer/sec, latency 3590 usec
```

Тренд положительный(`troughput` увеличился, `latency` уменьшился), поэтому
попробуем запустить с примитивным парметром `count: 4`:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 749.875 infer/sec, latency 1332 usec
Concurrency: 2, throughput: 1690.5 infer/sec, latency 1182 usec
Concurrency: 3, throughput: 1878.14 infer/sec, latency 1596 usec
Concurrency: 4, throughput: 2100.84 infer/sec, latency 1903 usec
Concurrency: 5, throughput: 2365.73 infer/sec, latency 2113 usec
Concurrency: 6, throughput: 2554.16 infer/sec, latency 2348 usec
Concurrency: 7, throughput: 2675.1 infer/sec, latency 2615 usec
Concurrency: 8, throughput: 2552.12 infer/sec, latency 3134 usec
```

Опять положительное изменение, но также появились Warnings:

```bash
[WARNING] Perf Analyzer is not able to keep up with the desired load. The results may not be accurate.
```

Значит, лучше поднять число vCPU до минимального возможного не вызывающего
ошибки перегрузки устройств. После подборов таким оказался `count: 5`:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 700.776 infer/sec, latency 1426 usec
Concurrency: 2, throughput: 1384.85 infer/sec, latency 1443 usec
Concurrency: 3, throughput: 1844.99 infer/sec, latency 1625 usec
Concurrency: 4, throughput: 2009.32 infer/sec, latency 1989 usec
Concurrency: 5, throughput: 2228.66 infer/sec, latency 2241 usec
Concurrency: 6, throughput: 2531.61 infer/sec, latency 2369 usec
Concurrency: 7, throughput: 2567.41 infer/sec, latency 2726 usec
Concurrency: 8, throughput: 2708.1 infer/sec, latency 2954 usec
```

**Итого**: оптимальный параметр `instance_group`:

```yaml
instance_group [
    {
        count: 5
        kind: KIND_CPU
    }
]
```

### Dynamic batching

Попробуем оценить влияние задержки между запросами. То есть изменим параметр
`max_queue_delay_microseconds` на `100`:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 621.951 infer/sec, latency 1607 usec
Concurrency: 2, throughput: 1485.43 infer/sec, latency 1346 usec
Concurrency: 3, throughput: 1579.86 infer/sec, latency 1897 usec
Concurrency: 4, throughput: 1762.79 infer/sec, latency 2268 usec
Concurrency: 5, throughput: 2140.81 infer/sec, latency 2333 usec
Concurrency: 6, throughput: 2159.52 infer/sec, latency 2778 usec
Concurrency: 7, throughput: 2255.4 infer/sec, latency 3102 usec
Concurrency: 8, throughput: 2373.65 infer/sec, latency 3370 usec
```

Результат стал хуже, попробуем значительно увеличить параметр до `4000`

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 621.951 infer/sec, latency 1607 usec
Concurrency: 2, throughput: 1485.43 infer/sec, latency 1346 usec
Concurrency: 3, throughput: 1579.86 infer/sec, latency 1897 usec
Concurrency: 4, throughput: 1762.79 infer/sec, latency 2268 usec
Concurrency: 5, throughput: 2140.81 infer/sec, latency 2333 usec
Concurrency: 6, throughput: 2159.52 infer/sec, latency 2778 usec
Concurrency: 7, throughput: 2255.4 infer/sec, latency 3102 usec
Concurrency: 8, throughput: 2373.65 infer/sec, latency 3370 usec
```

Видно, что результат лучше, но не на много, значит, мы прошли точку оптимума,
попробуем параметр `2000`:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 634.157 infer/sec, latency 1576 usec
Concurrency: 2, throughput: 1555.46 infer/sec, latency 1284 usec
Concurrency: 3, throughput: 1568.94 infer/sec, latency 1911 usec
Concurrency: 4, throughput: 1745.23 infer/sec, latency 2291 usec
Concurrency: 5, throughput: 1971.03 infer/sec, latency 2537 usec
Concurrency: 6, throughput: 2294.23 infer/sec, latency 2613 usec
Concurrency: 7, throughput: 2478.38 infer/sec, latency 2824 usec
Concurrency: 8, throughput: 2280.29 infer/sec, latency 3507 usec
```

Попробуем значение `1000`:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 658.984 infer/sec, latency 1516 usec
Concurrency: 2, throughput: 1333.69 infer/sec, latency 1498 usec
Concurrency: 3, throughput: 1707.94 infer/sec, latency 1754 usec
Concurrency: 4, throughput: 1786.17 infer/sec, latency 2239 usec
Concurrency: 5, throughput: 2019.08 infer/sec, latency 2476 usec
Concurrency: 6, throughput: 2180.27 infer/sec, latency 2751 usec
Concurrency: 7, throughput: 2341.14 infer/sec, latency 2988 usec
Concurrency: 8, throughput: 2401.34 infer/sec, latency 3329 usec
```

Значения лучше, но все еще недостаточно, попробуем рассмотреть `500`:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 634.334 infer/sec, latency 1575 usec
Concurrency: 2, throughput: 1425.86 infer/sec, latency 1402 usec
Concurrency: 3, throughput: 1647.03 infer/sec, latency 1819 usec
Concurrency: 4, throughput: 1879.14 infer/sec, latency 2128 usec
Concurrency: 5, throughput: 1991.57 infer/sec, latency 2510 usec
Concurrency: 6, throughput: 2145.81 infer/sec, latency 2796 usec
Concurrency: 7, throughput: 2307.51 infer/sec, latency 3030 usec
Concurrency: 8, throughput: 2400.12 infer/sec, latency 3334 usec
```

Скорость не изменилась, попробуем параметр `200`:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 687.719 infer/sec, latency 1453 usec
Concurrency: 2, throughput: 1464.06 infer/sec, latency 1365 usec
Concurrency: 3, throughput: 1734.08 infer/sec, latency 1728 usec
Concurrency: 4, throughput: 1868.24 infer/sec, latency 2140 usec
Concurrency: 5, throughput: 1987.72 infer/sec, latency 2516 usec
Concurrency: 6, throughput: 2441.36 infer/sec, latency 2456 usec
Concurrency: 7, throughput: 2440.98 infer/sec, latency 2868 usec
Concurrency: 8, throughput: 2535.91 infer/sec, latency 3153 usec
```

Видно, что параметр чем ближе к `100`, тем лучше значения, поэтому рассмотрим
вариант с `150`:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 656.16 infer/sec, latency 1523 usec
Concurrency: 2, throughput: 1371.09 infer/sec, latency 1459 usec
Concurrency: 3, throughput: 1652.49 infer/sec, latency 1815 usec
Concurrency: 4, throughput: 1893.37 infer/sec, latency 2111 usec
Concurrency: 5, throughput: 2180.52 infer/sec, latency 2291 usec
Concurrency: 6, throughput: 2466.73 infer/sec, latency 2432 usec
Concurrency: 7, throughput: 2552.49 infer/sec, latency 2742 usec
Concurrency: 8, throughput: 2719.33 infer/sec, latency 2939 usec
```

**Итого**: оптимальный параметр `dynamic_batching`:

```yaml
dynamic_batching: { max_queue_delay_microseconds: 150 }
```

## Арифметическая интенсивность

- GPU: RTX 3070 ti 8Gb
- [Specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-3070-ti.c3675):
  - Perfomance(FP32 (float)): 21.75 TFLOPS
  - Memory Bandwidth: 608.3 GB/s

Тогда $\frac{\#ops}{\#bytes} = \frac{21.75}{0.6083} \simeq 35.75$

Посчитаем через [thop](https://github.com/Lyken17/pytorch-OpCounter/) параметры
модели:

```bash
poetry run python3 perf_check.py
```

Получим следующий вывод для `BATCH_SIZE = 12`:

```bash
The currently activated Python version 3.10.6 is not supported by the project (^3.11).
Trying to find and use a compatible version.
Using python3.11 (3.11.7)
Device cpu
Loading model...
Perf measure...
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
[INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
OPS: 84.686M, BYTES: 2.374M #ops/#bytes = 35.66800208905437
```

и такой для `BATCH_SIZE = 13`:

```bash
The currently activated Python version 3.10.6 is not supported by the project (^3.11).
Trying to find and use a compatible version.
Using python3.11 (3.11.7)
Device cpu
Loading model...
Perf measure...
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
[INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
OPS: 91.743M, BYTES: 2.377M #ops/#bytes = 38.58871667866829
```

Итого для лучшей реализации GPU нужно использовать `BATCH_SIZE=13`.
