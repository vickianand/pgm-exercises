import numpy as np
from matplotlib import pyplot as plt

import argparse
from alpha_beta import q2_initialize, alpha_recur_stable, beta_recur_stable
from hmm_em import hmm_gaussian


def plot_smoothing(distr, lim, fname, show=False):
    k = distr.shape[1]
    f, ax = plt.subplots(k, 1, sharex=True)
    for i in range(k):
        ax[i].plot(range(lim), distr[:lim, i])

    plt.savefig(fname)
    if show:
        plt.show()
    plt.close()


def q2(test_data):
    mu, sig, pi, A = q2_initialize()

    alpha_norm, _ = alpha_recur_stable(test_data, pi, A, mu, sig)
    beta_norm, _ = beta_recur_stable(test_data, A, mu, sig)

    # smoothing-distribution calculation from alpha_norm and beta_norm
    log_smooting_distr = (
        alpha_norm
        + beta_norm
        - np.log(np.sum(np.exp(alpha_norm + beta_norm), axis=1)).reshape(-1, 1)
    )

    plot_smoothing(
        np.exp(log_smooting_distr), 100, fname="plots/Q2_smoothing_100.png", show=False
    )


def plot_clusters(data, hmm_model, title, fname, show=True):
    """ plots the points as per the viterbi decoding and also the cluster centers
    """
    viterbi_decoding = hmm_model.viterbi_decode(train_data)
    colors = ["m", "g", "r", "y"]
    for i in range(4):
        plt.scatter(
            train_data[viterbi_decoding == i, 0],
            train_data[viterbi_decoding == i, 1],
            marker=".",
            c=colors[i],
            alpha=0.5,
        )
        plt.scatter(
            hmm_model.mus[i][0],
            hmm_model.mus[i][1],
            marker="v",
            c=colors[i],
            s=70,
            edgecolors="k",
            label="initial center {}".format(i),
        )
    plt.title(title)
    plt.savefig(fname)
    if show:
        plt.show()
    plt.close()


def q9_and_10(test_data, hmm_model):
    mu, sig, pi, A = hmm_model.mus, hmm_model.sigmas, hmm_model.pi, hmm_model.A

    alpha_norm, _ = alpha_recur_stable(test_data, pi, A, mu, sig)
    beta_norm, _ = beta_recur_stable(test_data, A, mu, sig)

    # smoothing-distribution calculation from alpha_norm and beta_norm
    log_smooting_distr = (
        alpha_norm
        + beta_norm
        - np.log(np.sum(np.exp(alpha_norm + beta_norm), axis=1)).reshape(-1, 1)
    )

    plot_smoothing(
        np.exp(log_smooting_distr), 100, fname="plots/Q9_smoothing_100.png", show=False
    )

    most_likely = np.argmax(log_smooting_distr, axis=1)
    plt.plot(most_likely[:100])
    plt.savefig("plots/Q10_ml_state_100.png")
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--show")
    args = parser.parse_args()

    # load data
    train_data = np.loadtxt("data/EMGaussian.train")
    test_data = np.loadtxt("data/EMGaussian.test")

    # Q.2
    q2(test_data)
    print("-------------- Q2 done --------------")

    # # Q.4
    hmm_q4 = hmm_gaussian()
    hmm_q4.run_em(train_data, test_data)
    print("-------------- Q4 done --------------")

    plt.plot(hmm_q4.train_llhs, label="Training log-likelihoods")
    plt.plot(hmm_q4.test_llhs, label="Test log-likelihoods")
    plt.legend()
    plt.savefig("plots/Q5_log_llh_train_and_test.png")
    plt.show()
    plt.close()
    print("-------------- Q5 done --------------")

    # Q.8
    viterbi_decoding_train = hmm_q4.viterbi_decode(train_data)
    # print(viterbi_decoding_train)
    # plot for Q.8
    plot_clusters(
        train_data,
        hmm_q4,
        title="Clustering training points as per viterbi decoding",
        fname="plots/Q8_viterbi_cluster_train.png",
        show=False,
    )
    print("-------------- Q8 done --------------")

    # Q.9 & Q.10
    q9_and_10(test_data, hmm_q4)
    print("-------------- Q9 done --------------")
    print("-------------- Q10 done --------------")

    # Q.11
    viterbi_decoding_test = hmm_q4.viterbi_decode(test_data)
    # print(viterbi_decoding_test)
    # plot for Q.11
    plot_clusters(
        test_data,
        hmm_q4,
        title="Clustering test points as per viterbi decoding",
        fname="plots/Viterbi_cluster_test.png",
        show=False,
    )

    plt.plot(viterbi_decoding_test[:100])
    plt.savefig("plots/Q11_viterbi_state_100.png")
    # plt.show()
    plt.close()
    print("-------------- Q11 done --------------")
