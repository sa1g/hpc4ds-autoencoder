# hpc4ds-autoencoder

A C++20 image autoencoder developed for a High Performance Computing project. The model, training loop, neural-network layers, optimizer, dataloader, tests, and benchmarks are implemented around Eigen, with support for sequential, OpenMP, MPI, and hybrid MPI + OpenMP execution.

The project currently targets grayscale 28x28 image datasets organized by class, with support for MNIST and preprocessed SVHN.

This project was implemented in the context of the High Performance Computing for Data Science offered by prof. Fiore @ UniTrento.

## Features

- Fully connected autoencoder: `784 -> 256 -> 784`
- Custom `Linear`, `ReLU`, `MSE`, and SGD implementations
- PNG dataloader for class-structured image folders
- Sequential, OpenMP, MPI, and hybrid MPI + OpenMP builds
- MPI data parallelism with per-rank data sharding and weight averaging
- TensorBoard logging under `runs/`
- Unit tests with GoogleTest
- Micro and communication benchmarks with Google Benchmark
- Singularity/Apptainer and PBS cluster scripts

## Project Layout

```text
.
├── main.cc                 # Experiment entry point
├── src/
│   ├── data/               # PNG dataloader
│   ├── model/              # AutoencoderModel
│   ├── nn/                 # Linear, ReLU, Sigmoid, MSE
│   ├── optim/              # SGD
│   └── utils/              # Training loop, MPI worker, utilities
├── tests/                  # GoogleTest tests
├── benchmarks/             # Google Benchmark executables
├── data/                   # Dataset scripts and generated datasets
├── scripts/                # Build, run, benchmark, and PBS scripts
├── report/                 # Typst report and PDF
├── singularity.def         # Container definition
└── CMakeLists.txt
```

## Requirements

The recommended environment is Singularity/Apptainer.

The container installs the C++ toolchain, CMake, OpenMPI 4.1.6, Protobuf, Python, and system dependencies. CMake downloads Eigen, stb, GoogleTest, Google Benchmark, and `tensorboard_logger` through `FetchContent`.

For SVHN preprocessing, the Python requirements are:

```text
scipy
numpy
pillow
tqdm
```

## Build the Container

```bash
apptainer build singularity.sif singularity.def
# or
singularity build singularity.sif singularity.def
```

Building the container can take some time because OpenMPI and Protobuf are compiled from source.

## Dataset Preparation

The executable expects datasets in this format:

```text
data/<dataset>/
├── train/
│   ├── 0/
│   ├── 1/
│   └── ...
└── test/
    ├── 0/
    ├── 1/
    └── ...
```

Each image must be a grayscale 28x28 PNG.

Download and prepare MNIST:

```bash
make prepare_mnist
```

Prepare both MNIST and SVHN:

```bash
make all
```

Remove generated datasets:

```bash
make clean_data
```

SVHN is downloaded as `.mat`, converted to grayscale 28x28 PNG images, and written under `data/svhn`.

## Build

The dataset is selected at configure time with `DATASET_NAME`. The default is `mnist`.

Main CMake options:

| Option | Description |
|---|---|
| `-DDATASET_NAME=mnist` | Use `data/mnist` |
| `-DDATASET_NAME=svhn` | Use `data/svhn` |
| `-DWITH_OPENMP=ON` | Enable OpenMP |
| `-DWITH_MPI=ON` | Enable MPI |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build |

Build all execution variants locally through the container:

```bash
DATASET_NAME=mnist bash scripts/build_local.sh
```

This creates:

```text
build/build_seq_mnist
build/build_omp_mnist
build/build_mpi_mnist
build/build_hybrid_mnist
```

Manual hybrid build example:

```bash
singularity exec singularity.sif cmake -S . -B build/build_hybrid_mnist \
  -DCMAKE_BUILD_TYPE=Release \
  -DDATASET_NAME=mnist \
  -DWITH_MPI=ON \
  -DWITH_OPENMP=ON

singularity exec singularity.sif cmake --build build/build_hybrid_mnist -j
```

## Run

Sequential:

```bash
singularity exec singularity.sif ./build/build_seq_mnist/autoencoder
```

OpenMP:

```bash
OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMP_PLACES=cores \
singularity exec singularity.sif ./build/build_omp_mnist/autoencoder
```

MPI:

```bash
mpirun -np 4 \
  singularity exec -B "$PWD" singularity.sif \
  ./build/build_mpi_mnist/autoencoder
```

Hybrid MPI + OpenMP:

```bash
OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores \
mpirun -np 4 --bind-to none \
  singularity exec -B "$PWD" singularity.sif \
  ./build/build_hybrid_mnist/autoencoder
```

The current experiment configuration is defined in `main.cc`:

```text
batch_size = 256
input_dim  = 28 * 28
hidden_dim = 256
output_dim = 28 * 28
lr         = 0.01
epochs     = 20
```

## Output and TensorBoard

At startup, the program prints the dataset, execution mode, number of epochs, batch size, learning rate, world size, and output path.

TensorBoard logs are written to:

```text
runs/<dataset>_<mode>_<timestamp>/tfevents.pb
```

Logged metrics include:

- `train_loss`
- `eval_loss`
- `epoch_time_sec`
- `samples_per_sec`
- `test_loss`
- `total_time_sec`
- `total_samples_per_sec`

View logs with:

```bash
tensorboard --logdir runs
```

In MPI runs, only rank 0 writes TensorBoard events. Losses are averaged across ranks, epoch time uses the slowest rank, and throughput uses the total number of training samples processed across ranks.

## Tests

Run all tests:

```bash
ctest --test-dir build/build_seq_mnist --output-on-failure
```

Run only project tests:

```bash
ctest --test-dir build/build_seq_mnist -L autoencoder_tests --output-on-failure
```

## Benchmarks

Micro-benchmarks are built when `WITH_OPENMP=ON`. The communication benchmark is built when `WITH_MPI=ON`.

Example micro-benchmark:

```bash
singularity exec singularity.sif \
  ./build/build_hybrid_mnist/benchmarks/b_linear
```

MPI communication benchmark:

```bash
mpirun -np 4 \
  singularity exec -B "$PWD" singularity.sif \
  ./build/build_hybrid_mnist/benchmarks/b_communication_many_calls
```

Write benchmark output as JSON:

```bash
./build/build_hybrid_mnist/benchmarks/b_linear \
  --benchmark_out=logs/b_linear.json \
  --benchmark_out_format=json
```

## PBS Cluster Usage

The scripts in `scripts/` assume a PBS cluster, the `shortCPUQ` queue, Singularity, and the module:

```text
OpenMPI/4.1.6-GCC-13.2.0
```

Adjust these values for the target cluster.

Build all variants on the cluster:

```bash
qsub -v DATASET_NAME=mnist scripts/build.sh
```

Prepare datasets:

```bash
qsub scripts/data_get_and_preprocess.sh
```

Launch MPI runs:

```bash
bash scripts/launch_mpi.sh
```

Launch hybrid MPI + OpenMP runs:

```bash
bash scripts/launch_hybrid.sh
```

Launch communication benchmarks:

```bash
bash scripts/launch_comm_bench.sh
```

Launch micro-benchmarks:

```bash
qsub -v DATASET_NAME=mnist scripts/micro_bench.sh
```

Logs and benchmark JSON files are written under `logs/`.

## Implementation Notes

- The training set is split into train/evaluation subsets using a fixed seed.
- In MPI mode, each rank receives a shard of the training data.
- Evaluation and test data are kept identical across ranks.
- MPI model synchronization averages weights across ranks after each epoch.
- The dataloader skips the final partial batch if it is smaller than `batch_size`.
- Generated build directories, datasets, TensorBoard runs, logs, and `.sif` images are ignored by git.

## License

MIT.