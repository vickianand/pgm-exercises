Expectation Maximization algorithm and Gaussian Mixtures
========================================================

The fileEMGaussian.train in `hwk3data` contains samples of data xi where xi ∈ R^2 (one datapoint per row).  The goal of this exercise is to implement the EM algorithm for some mixtures of KGaussians in R^d (here d= 2 and K= 4), for i.i.d. data.  

Implement the K-means algorithm 
--------------------------------

Represent graphically the training data, the cluster centers, as well as the different clusters (use 4 colors). Try several random initializations and compare results (centers and the actual K-means objective values).

EM algorithm for GMM with isotropic coviances
---------------------------------------------

Use the clustering given by K-means as the initialization.
Represent graphically the training data, the centers, as well as the covariance matrices (an elegant way is to represent the ellipse that contains a specific percentage, e.g., 90%,of the mass of the Gaussian distribution).
Estimate and represent (with different colors or different symbols) the most likely latent variables for all data points (with the parameters learned by EM).

EM algorithm for GMM with general coviances
---------------------------------------------

Do same graphical representations and estimation done for previous GMM with isotropic covariance.

-------------------------------------------------------------------------------

* All the code is written in a single jupyter notebook.

* The code is written for python 3.

* It requires numpy and matplotlib libraries.(base)

* [Look at question 8 of the homework](https://1drv.ms/b/s!As-tXka0OqUohONqgdjx_Is8T-k8DQ)
