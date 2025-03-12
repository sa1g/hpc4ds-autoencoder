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

# About unit-testing
## Notes from this [speech](https://www.youtube.com/watch?v=fr1E9aVnBxw)
Verify that a kknwon, fixed input procudes a known, fixed output.

Eliminate everything that makes input/output unclear or contingent
 - Never generate random input, always use fixed values
 - Don't use named constants from the model code, they may be wrong or change. Prefer literal string and numbers.
 - Don't access the network and preferably not hte file system

1. Write the tests first!
    - Writing the model code first is slowing you down and won't get you to a good test coverage
    - It's not about tewsting, it's about software development
    - Test first development creates better API because you start with the user, not the used
    - Test first hides implementation and avoids exposing internal implementation defails. It avoids brittle, tightly coupled tests.
        - So if you wrote the tests before the implementation then they are only coupled to the API, not the implementation, so you can change the implementation without changing the tests.
1. When writing a test
    - Make sure the test fails before writing a test that passes
        - This is to be sure that the test is actually testing something
1. Why unit tests?
    - unit means one. Each test tests exactly one thing.
    - Each test method is one test
    - Best practice: one assert per one test
    - share setup in a fixture, not hte same mothod
    - you can have multiple test classes per model class. Do no feel compelled to stuff all your tests for `Foo` in `FooTest`
1. Unit also means Independent
    - Tests can (and do) run i nany order
    - Tests can (and do) run in parallel in multiple threads
    - Tests should not interfere with each other
1. Tests nad Thread Safety
    - Don't use synchronization, semaphores or special data structures in tests
    - Don't share data between tests
        - do not use non-constant static fields in your tets
        - Be wary of global state in teh model code under test
    - Share setup in a fixture, no the same method
    - Tests do not share instance data
    - Every test that needs a slightly different setup cna go into a seprate test class
1. Speed
    - This is for ease of development
    - A single test shoudl run in a second or less
    - A complete suite should run in a minute or less
    - Separate larger tests into additional seuites
    - Fail fast. Run slower tests last.
1. Failing test should produce clear output
    - shoudl give clear, unambiguous erorr messages
    - rotate test data:
        - don't use the same data in every test
        - easiert to see immediately which test is failing and why
1. Flakiness
    - when a test passes and doesn't passes without any code change
    - possible sources
        - time dependence (e.g. guis)
        - network availability
        - explicit randomness
        - multithreading
    - System skew
        - it runs on my pc but not on yours
        - possible sources
            - multithreading
            - assumptions about the OS
            - undefined behavior
                - floating point roundoff
                - integer width
                - default character set
                - ecc.
1. Avoid conidtional logc in tests 
    - Do not put if statements in tests
1. Debugging
    - Write a failing test before you fix the bug.
    - if the test passes, the bug isn't what you think it is
1. Refactoring
    - Break the code before you refactor it.
        1. Verify that there are tests for the code you are about to refactor
        1. Run the tests
        1. Refactor
    - Check the code coverage
    - If necessary, write additional tests before doing unsafe refactorings.
1. Development Practices
    - Use continuous integration (e.g. Travis)
    - Use a submit queue
    - Never, ever allow a check in with a failing test
    - IF it happens, rollback first; ask qwuestions later.
    - A red test blocks all merges. No further check ins until the build is green.
1. Finally
    - write tests first
    - make all test unambiguous and reproducable


