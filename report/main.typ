#import "template.typ": ieee

#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  body
}

#show: ieee.with(
  title: [Parallel Autoencoder],
  subtitle: [High Performance Computing for Data Science],
  abstract: [
    TODO
  ],
  authors: (
    (
      name: "Silvanus Bordignon",
      department: [xxxxxx],
      organization: [Univesity of Trento],
      location: [Trento, Italy],
      email: "maedje@typst.app"
    ),
    (
      name: "Ettore Sasggiorato",
      department: [247178],
      organization: [University of Trento],
      location: [Trento, Italy],
      email: "ettore.saggiorato@studenti.unitn.it"
    ),
  ),
  index-terms: ("HighPerformanceComputing", "DeepLearning"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
  appendix: [
    =
    "asd", adwaoids
     = If needed we can add a title to the appendix :)
- Ablations about where perofmrances explode + some thougts about the design process
- heap vs stack with Eigen (`Matrix` `<float, ...>` vs `MatrixXf`)
- using vs not using eigen parallelization: it's parallelization (single loop) on a set of data vs a possible openmp done by us where data is worked on in parlalel
    
    
    ],
)

= Introduction
Aim of the project.
Introduce that we use a small neural network, so the network won't be split between multiple nodes.

== Instruction for Reproducibility and Building
cmake. runs on linux. Needs GCC and C++20.

= State of the Art / Related Works

= Libraries and Datasets
== Libraries
- stb
- Eigen
- Gtest
- GBench

== Datasets


= Methodology and Implementation Details
The structure of the operations of nn is fundamental, our objective is to maximize the TOPS. 
- Critical thought about the cost of allocating memory and the usage of the heap/stack -> reference to the appendix
- Considering that Eigen can parallelize the workload thoughts about avoiding too much context switch on the CPU -> reference appendix
- Difference between when one has a GPU and doesn't in the dataloader
- Why we are doing distributed training instead of e.g. sharing the model between machines (Efficiency of course :) ). Reference a few papers plz

We planned to work sequentially: 
1. Basic implementation single threaded
2. Activating Eigen's OpenMP parallelization
3. OpenMP the rest of hte code
4. MPI -> check how to use infiniband/omnipath 
5. MPI + OpenMP 

Say that methods are benchmarked individually to see where we gain the best perforamance and where it becomes worse.

== Unit Testing
Performed to be sure that modules work correcly and that when parallelizing/using mpi nothing breaks.

== Basic Implementation
- Why many parts of the NN are set as a template library.
- APIs inspired from PyTorch python's apis.

-------------------------------------

- Dataloader
- Linear layer
- ReLU
- Sigmoid
- Encoder/Decoder
- Loss
- Backpropagation -> reference from the book (need to get it lol)
- Gradient Descent

== Eigen parallelization
== OpenMP
== MPI
> How we are parallelizing on multiple nodes
> What we are sharing and Why
> ABLATION on the distributed weights
== Combo: MPI + OpenMP

= System Description

= Experiments
== Evaluation
What we evaluate:
- Speedup
- Efficiency
- Scalability
  - Strong
  - Weak

=== What we expect

== TODO: experiment combos and their results compared.
What we can learn from this, how it goes.


= Tables and Data <app1>
= Additional Listings <app2>
