Until the project is finished all notes **will not** be formal at all.

# Project Objective
The objective of this project is to develop an Image AutoEncoder from scratch and benchmark it to show the potentiality of parallelizing it in an HPC scenario (without dedicated GPUs).

> Why image autoencoders? Because they are the easiest to do and this project is not focused on AI, but on the HPC part.

# What we need to develop
1. Dataset/Dataloader
    - **Dataset class**: load and pre-process images (e.g. normalization, reshape). Possibly flatten images into vectors of pixels.
    - **Dataloader**: read data in batches
        - Read (e.g. CSV file) file names
        - Load images in batches
        - Shuffle images (for the training part)
    - DataSet splitter for train/test/eval
2. Autoencoder
    - **Linear Layer** - example
        ```c++
        class Linear {
        public:
            Eigen::MatrixXf weights;
            Eigen::VectorXf biases;

            Linear(int input_size, int output_size) {
                weights = Eigen::MatrixXf::Random(input_size, output_size);
                biases = Eigen::VectorXf::Zero(output_size);
            }

            Eigen::VectorXf forward(const Eigen::VectorXf& input) {
                return (weights.transpose() * input + biases);
            }
        };
        ```
    - **Relu** - example
        ```c++
        Eigen::VectorXf relu(const Eigen::VectorXf& input) {
            return input.cwiseMax(0);
        }
        ```
    - **Sigmoid** - example
        ```c++
        Eigen::VectorXf sigmoid(const Eigen::VectorXf& input) {
            return input.array().exp() / (input.array().exp() + 1);
        }
        ```
    - **Encoder-Decoder setup**
        ```c++
        class Autoencoder {
        public:
            Linear encoder1, encoder2, decoder1, decoder2;
            
            Autoencoder(int input_size, int latent_size) 
                : encoder1(input_size, 128), encoder2(128, latent_size), 
                decoder1(latent_size, 128), decoder2(128, input_size) {}

            Eigen::VectorXf forward(const Eigen::VectorXf& input) {
                Eigen::VectorXf encoded = relu(encoder1.forward(input));
                encoded = relu(encoder2.forward(encoded));
                
                Eigen::VectorXf decoded = relu(decoder1.forward(encoded));
                return sigmoid(decoder2.forward(decoded));  // Output reconstruction
            }
        };
        ```
    - **Loss**: MeanSquaredError
        ```c++
        float mse_loss(const Eigen::VectorXf& input, const Eigen::VectorXf& output) {
            return (input - output).squaredNorm() / input.size();
        }
        ```
    - **Backpropagation and Gradient Descent**
3. Training
    - Training loop
    - Testing loop
    - Eval loop
    - Logging

# Datasets
- MNIST
    - handwritten numbers between 0 and 9
    - 60k images
    - black and white
    - good for fast tests

    - we got the [png version](https://github.com/myleott/mnist_png/tree/master), note that it's not the original one, but currently the [official website](http://yann.lecun.com/exdb/mnist/) redirects to nothing.

- SVHN
    - street view house numbers
    - 600k images
    - rgb
    - good to test (load) more complex architecture setups

    - got it form the [official website](http://ufldl.stanford.edu/housenumbers/)

# Parallelism
## About the model
- Given the datasets the model won't be so big that model parallelism would be needed.
- Focus on **Data Parallelism**
## About HPC
Bench all the different methods so taht we can compare efficiency and throughput individually
- simple single node
- open_mp
- mpi
- hybrid mpi+open_mp

- We may also test with both `central` and `distributed` nodes sync steup.
    - It may be interesting to see which is faster.
    - I suppose that the distributed
        - issue: broadcast n-nodes to n-nodes
        - pro: no need to send an answer

# Libraries
## Linear Algebra (BLAS)
I was thinking of using common, free and opensource, not hardware dependant, and not HPC specific libraries. 
- Eigen: well known, fast
- Fastor: newer, may be faster, need to check

## HPC
- openmp
- mpi

## Benchmarks, Memory Check and Behavior Analysis
- google benchmark
- Valgrind
- Cppcheck

## Build Tools
Good old 
- CMAKE - [Tutorial](https://github.com/ttroy50/cmake-examples)
https://cliutils.gitlab.io/modern-cmake/chapters/intro/dodonot.html 
It's installed on the university HPC, we don't have to worry about this.

# Notes
`cmake .. -DCMAKE_BUILD_TYPE=Debug` to build in debug mode
