Goal of the project : investigate the dynamics of learning of a Kernel Regression Setting in a Teacher/Student framework

Teacher/Student Regression framework for supervised learning :

Some data is generated according to a given distribution : here the Teacher generates points $Z_T \sim \mathcal{N}(\vec{0},K_T)$ where $K_T(x,x') = K_T(||x - x'||)$ is the so-called Teacher kernel. These points constitute the Training Set. From it, the Student (characterized by an isotropic student kernel $K_S(x,x') = K_S(||x - x'||)$) tries to infer the underlying law. Its performance is measured by the test error $\epsilon$, defined as a quadratic loss between the prediction of the Student and true values at several *new/unseen* points generated by the Teacher.

We investigate the dynamics of the test error during the optimization routine explained hereafter, guided by the knowledge of the behavior of the exact test error ($\approx$ test error at infinite time of optimization) developed in
??? <quote>

In this project, the Teacher kernel is chosen among the Matérn kernels family :
C_\nu(d) = \sigma^2\frac{2^{1-\nu}}{\Gamma(\nu)}\Bigg(\sqrt{2\nu}\frac{d}{\rho}\Bigg)^\nu K_\nu\Bigg(\sqrt{2\nu}\frac{d}{\rho}\Bigg),
where $\Gamma$ is the [[gamma function]], $K_\nu$ is the modified [[Bessel function]] of the second kind, ''\rho=1''is the characteristic length scale and ''\nu'' is the non-negative smoothness parameter of the covariance (the larger the smoother, \nu = \infinity <=> Gaussian kernel)

and the Student kernel is the Laplace kernel <=> Matérn[\nu = 1/2 ; \rho = 1] : K_S(x,x') = exp(-||x - x'||)

Optimization routines :

The prediction of the student is computed within the `predict` method : the prediction at a given point X is a simple weighted average of similarities between X and all the training points. The weights are initially zeros but they are optimized during the training.

Three optimization Algorithms are implemented : `GradientDescent` (GD), `ConjugateGradientDescent` (CGD) and `ConjugateGradientDescentFixedEpochs`
* GD : the easiest to implement, easy to work out analytically but very slow numerically. Fixed learning rate, rescaled by the number of training points $P$.
* CGD : more complicated to work out analytically but very efficient numerically. Converges in approximately $P$ epochs. It comes in two versions :
    * The default version, where the algorithm stops when the train loss reaches a given threshold.
    * The benchmark version, where the algorithm runs for a fixed number of epochs passed in argument. Also useful when one is interested in the beginning of the dynamics.

A few remarks :
* I recommend no preconditioning since it modifies greatly the results, leading to false conclusions.
* The code is designed to collect statistics by running independent realisations to emulate the expectation over the Teacher random process. A brute-force approach would be to run all simulations the same number of times but it would take way too long. Therefore, since at large \nu and at large P (independently), the standard deviation of the results goes to zeros, one concentrates the efforts (= more realisations) for small \nu and small P
