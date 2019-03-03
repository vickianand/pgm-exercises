import numpy as np
from matplotlib import pyplot as plt


def q2_initialize():
    mu1 = np.array([-2.0344, 4.1726])
    mu2 = np.array([3.9779, 3.7735])
    mu3 = np.array([3.8007, -3.7972])
    mu4 = np.array([-3.0620, -3.5345])

    sig1 = np.array([2.9044, 0.2066, 0.2066, 2.7562]).reshape(2, 2)
    sig2 = np.array([0.2104, 0.2904, 0.2904, 12.2392]).reshape(2, 2)
    sig3 = np.array([0.9213, 0.0574, 0.0574, 1.8660]).reshape(2, 2)
    sig4 = np.array([6.2414, 6.0502, 6.0502, 6.1825]).reshape(2, 2)

    mu = np.array([mu1, mu2, mu3, mu4])
    sig = np.array([sig1, sig2, sig3, sig4])
    # print("mu.shape: ", mu.shape, ", sig.shape: ", sig.shape)

    pi = np.array([0.25, 0.25, 0.25, 0.25])
    A = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            A[i][j] = 0.5 if i == j else 1.0 / 6.0

    # print('pi: ', pi, '\n', 'A: ', A)
    return mu, sig, pi, A


# mu, sig, pi, A = q2_initialize()
def gaussian(x, mu, sigma):
    """
    x : d
    mu : d
    sigma : d X d
    """
    g = 1 / np.sqrt(np.linalg.det(2 * np.pi * sigma))
    g *= np.exp(-0.5 * (x - mu).T.dot(np.linalg.inv(sigma)).dot(x - mu))
    return g


# print(train_data[0], mu[0], sig[0])
# for j in range(4):
#     print(gaussian(test_data[0], mu[j], sig[j]))


def k_gaussians(x, mu, sigma):
    """
    x : d
    mu : k X d
    sigma : k X d X d 
    """
    g = 1 / np.sqrt(np.linalg.det(2 * np.pi * sigma))  # .reshape(-1, 1) # k X 1
    x_c = x - mu  # k X d
    for i in range(sigma.shape[0]):
        g[i] *= np.exp(-0.5 * x_c[i].T.dot(np.linalg.inv(sigma[i])).dot(x_c[i]))
    return g.reshape(-1, 1)


def alpha_recur(X, pi, A, mu, sigma):
    k = pi.shape[0]
    n = X.shape[0]
    alpha = np.zeros((n, k))

    # initialization
    for j in range(k):
        alpha[0][j] = pi[j] * gaussian(X[0], mu[j], sigma[j])

    for t in range(1, n):
        o = k_gaussians(X[t], mu, sigma)
        alpha[t] = o.reshape(-1) * A.dot(alpha[t - 1])
    return alpha


def alpha_recur_stable(X, pi, A, mu, sigma):
    k = pi.shape[0]
    n = X.shape[0]

    alpha_norm = np.zeros((n, k))
    alpha_max = np.zeros(n)

    # initialization
    o = k_gaussians(X[0], mu, sigma)
    for j in range(k):
        alpha_norm[0][j] = np.log(pi[j] * o[j])

    alpha_max[0] = np.max(alpha_norm[0])
    alpha_norm[0] = alpha_norm[0] - alpha_max[0]

    for t in range(1, n):
        o = np.log(k_gaussians(X[t], mu, sigma))
        alpha_norm[t] = (
            o.reshape(-1) + np.log(A.dot(np.exp(alpha_norm[t - 1]))) + alpha_max[t - 1]
        )
        alpha_max[t] = np.max(alpha_norm[t])
        alpha_norm[t] -= alpha_max[t]

    return alpha_norm, alpha_max


def beta_recur_stable(X, A, mu, sigma):
    k = mu.shape[0]
    n = X.shape[0]

    beta_norm = np.zeros((n, k))
    beta_max = np.zeros(n)

    # initialization for recursion
    # beta_norm[0] = np.log(np.ones(k))
    # beta_max[0] = np.max(beta_norm[0])
    # beta_norm[0] = beta_norm[0] - beta_max[0]

    for t in range(n - 2, -1, -1):
        o = k_gaussians(X[t + 1], mu, sigma)
        beta_norm[t] = (
            np.log(
                A.dot(
                    ((o * np.exp((beta_norm[t + 1]).reshape(-1, 1))).reshape(-1, 1))
                ).reshape(-1)
            )
            + beta_max[t + 1]
        )
        beta_max[t] = np.max(beta_norm[t])
        beta_norm[t] -= beta_max[t]

    return beta_norm, beta_max

