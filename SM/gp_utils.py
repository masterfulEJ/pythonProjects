import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
#from numpy.linalg import inv

# Function to get the GP solution


def gp_solve(x_train, y_train, x_pred, kernel, sig_noise=0, **kwargs):
    k_xx = kernel(x_train, x_train, **kwargs)
    k_x_xp = kernel(x_train, x_pred, **kwargs)
    k_xp_x = kernel(x_pred, x_train, **kwargs)
    k_xp_xp = kernel(x_pred, x_pred, **kwargs)

    Vinv = np.linalg.inv(k_xx + (sig_noise**2) * np.identity(len(k_xx)))
    mu = np.dot(np.dot(k_xp_x, Vinv), y_train)
    var = k_xp_xp - np.dot(np.dot(k_xp_x, Vinv), k_x_xp)

    return mu, var

# function to plot the GP solution


def gp_plot(x, y, x_pred, mu_pred, cov_pred, n_sample=0, main=""):

    se_pred = 2 * np.sqrt(np.diag(cov_pred))
    # plot samples from the posterior
    # (this can be misleading, we don't actually have the function, we have draws from an MvN using the learned mean and covariance of the function)
    for i in range(n_sample):
        samp_y = np.squeeze(
            np.random.multivariate_normal(mu_pred, cov_pred, 1))
        plt.plot(x_pred, samp_y, 'red', alpha=0.3)
    # plot the mean
    plt.plot(x_pred, mu_pred, 'red', alpha=0.7)
    # plot the observations
    plt.plot(x, y, 'o', markersize=3, color='blue', alpha=0.5)
    # plot prediction uncertainty
    plt.fill_between(x_pred, mu_pred - se_pred, mu_pred +
                     se_pred, color='pink', alpha=0.7)
    plt.grid(alpha=0.2)
    plt.title(main)


if __name__ == '__main__':
    gp_solve()
    gp_plot()
