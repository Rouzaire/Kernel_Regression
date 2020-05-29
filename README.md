Ylann Rouzaire, all rights reserved.

# Kernel Regression Dynamics

## Goal of the project

Investigate the dynamics of learning of a Kernel Regression in a Teacher/Student framework

## Structure of the code

The project is small enough for the code to be organized as follows :
* All the functions are defined in `function_definitions.jl`
* The `main.jl` defines the keys arguments, among them the number of statistics to collect and distributes the work on different processors thanks to the function `Run`.
* The data is saved in JLD files byt the `Run` function so that the analysis can be performed later by the `analysis.jl` file.
* The `benchmark.jl` is a test file and therefore may contain deprecated syntax.

## Some remarks

* The Teacher/Student framework for supervised Regression is explained in details in the [`report`](#report.pdf)file.

* The kernels we use for Teachers are from the Matérn family : *[Read more](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)*. The Student kernel is kept fixed to Matérn[ν = 1/2]  (aka Laplace kernel or exponential kernel). These kernels are isotropic and translation invariant. They are coded in the function `k(h)` .

* The data is generated uniformly on the unit hypersphere of dimension d (for clarity : d=1 means the unit circle and d=2 means the usual sphere embedded in the natural 3 dimensions) by normalizing (to 1) points from a multivariate random normal distribution in d+1 dimensions. From this data one can then extract the training sets and testing sets. Therefore, to constructs theses sets, one has to call successively `generate_X(...)` ,  `generate_Z(...)` , `extract_TestSet(...)` , `extract_TrainSet(...)` .

* For prediction, the Student needs a Gram Matrix K defined as : (K)ij = k(x_i,x_j). This positive definite matrix is computed is the `ConstructGramMatrices` function. The pdness is mathematically guaranteed by the pdness of the Matérn kernels. However, some numerical issues sometimes imply that the matrix is not positive definite. Hence the `EnforcePDness` function that perturbs slightly the matrix by adding a tiny jitter on the diagonal.


## Optimization routines :

We investigate the dynamics of the test error during the optimization routine explained hereafter, guided by the knowledge of the behavior of the exact test error (≈ test error at infinite time of optimization) developed in *Learning Curves of Kernel Methods, empirical data vs. Teacher/Student paradigm* by Spigler, Geiger and Wyart.

The prediction of the student is computed within the `predict` method : the prediction at a given point X is a simple weighted average of similarities between X and all the training points. The weights are initially zeros but they are optimized during the training.

Three optimization Algorithms are implemented : `GradientDescent` (GD), `ConjugateGradientDescent` (CGD) and `ConjugateGradientDescentFixedEpochs`
* GD : the easiest to implement, easy to work out analytically but very slow numerically. Fixed learning rate η, rescaled by the number of training points P.
* CGD : more complicated to work out analytically but very efficient numerically. Converges in approximately P epochs. It comes in two versions :
    * The default version, where the algorithm stops when the train loss reaches a given threshold.
    * The benchmark version, where the algorithm runs for a fixed number of epochs passed in argument. Also useful when one is interested in the beginning of the dynamics.
* The leading term in runtime complexity is O(#number_epochs * P * max(P,Ptest))

### Additional remarks :
* I *strongly advise against preconditioning* (other than `EnforcePDness`) since it modifies greatly the results, leading to false conclusions.
* Surprisingly, adding momentum to GD did not accelerate convergence so it is not implemented in the current version.
* The code is designed to collect statistics by running independent realisations to emulate the expectation over the Teacher random process. A brute-force approach would be to run all simulations the same number of times but it would take way too long. Therefore, since at small ν and at large P (independently), the standard deviation of the results goes to zeros, one concentrates the efforts (= more realisations) for large ν and small P

## Bibliography
* Spigler, Geiger and Wyart : Learning Curves of Kernel Methods, empirical data vs. Teacher/Student paradigm.
* Jacot, Gabriel and Hongler : Neural Tangent Kernel Convergence and Generalization in Neural Networks
* Bordelon, Canatar and Pehlevan : Spectrum dependent learning curves in kernel regression and wide neural networks
