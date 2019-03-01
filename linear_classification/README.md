Linear classification
=====================

Here we have 6 text files: three training sets (.train files) and three test sets (.test files). Each row of the text file represents a sample of data (xi, yi).

There are three columns: the first two give the coordinates for xi ∈ R^2; the third column gives the class label yi ∈ {0, 1}. There are three different types of datasets (A, B and C), all generated from some kind of mixture of Gaussians generative model. The train and test sets are generated from the same distribution for each types of dataset. Here is the actual generating process:

* Dataset A: the class-conditionals for this dataset are Gaussians with different means, but with a shared covariance matrix Σ.

* Dataset B: similar generating process but the covariance matrices are different for the two classes.

* Dataset C: here one class is a mixture of two Gaussians, while the other class is a single Gaussian (with no sharing).

Note that normally we would not know the information about the generating process. In this
assignment, we will implement and compare difference classification approaches.

For all the below discussed models, we implement the maximum likelihood estimator and represent graphically the data as well as the decision boundary.

Generative model (Fisher LDA).
---------------------------------

We first consider the Fisher LDA model: given the class variable, the data are assumed to be Gaussians with different means for different classes but with the same covariance matrix.

Logistic regression
-------------------

We implement logistic regression for an affine function f(x) = wx + b using the IRLS algorithm.

Linear regression
-----------------

We can forget that the class y can only take the two values 0 or 1 and think of it as a real-valued variable on which we can do standard linear regression (least-squares). Here, the Gaussian noise model on y does not make any sense from a generative point of view; but we can still do least-squares to estimate the parameters of a linear decision boundary (its performance is surprisingly good despite coming from a “bad” generative model!).

QDA model
----------

We finally relax the assumption that the covariance matrices for the two classes are the same. So, given the class label, the data are now assumed to be Gaussian with means and covariance matrices which are a priori different:

Y ∼ Bernoulli(π)

X | Y = j ∼ N (µj, Σj)

-------------------------------------------------------------------------------

[Link to homework questions](https://1drv.ms/b/s!As-tXka0OqUohONnfySkvxFDEvwPFw)
