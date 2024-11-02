import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class SimReg():
    """
    Simulate data for a regression problem.
    """

    def __init__(self, n, k, constant=True, mean=None, cov=None, coefficients=None, **kwargs):
        self.n = n
        self.k = k
        self.constant = constant
        self.cov = cov
        self.kwargs = kwargs

        if not coefficients:
            self.coefficients = np.random.randn(k)
        else:
            self.coefficients = coefficients

        # generate feature data with some small covariance
        # Mean of the multivariate normal
        if not mean:
            self.mean = np.zeros(k)  
        else:
            self.mean = mean

        if not cov:
            # Covariance matrix
            cov = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    if i == j:
                        cov[i][j] = 1
                    else:
                        cov[i][j] = np.random.rand() * 0.1
            # Make the matrix symmetric
            for i in range(k):
                for j in range(k):
                    if i != j:
                        cov[i][j] = cov[j][i]
            self.cov = cov
        else:
            self.cov = cov

        X = multivariate_normal.rvs(self.mean, self.cov, size=self.n) 
        y = np.ones(n) + X @ self.coefficients + np.random.randn(self.n)
        X_corr = np.corrcoef(X, rowvar=False)

        self.X, self.y, self.X_corr = X, y, X_corr
    

    def plot_features(self):
        # plot y ~ X for each feature in a grid
        fig, axes = plt.subplots(int(np.ceil(np.sqrt(self.k))), int(np.floor(np.sqrt(self.k))), figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.scatter(self.X[:, i], self.y, alpha=0.1)
            ax.set_title(f"Feature {i}")
        plt.tight_layout()
        plt.show()

    def plot_distributions(self):
        # plot histograms of X's and y
        fig, axes = plt.subplots(1, self.k+1, figsize=(10, 2))
        for i, ax in enumerate(axes.flat):
            if i == self.k:
                ax.hist(self.y, bins=30)
                ax.set_title("y")
            else:
                ax.hist(self.X[:, i], bins=30)
                ax.set_title(f"Feature {i}")

        plt.tight_layout()
        plt.show()

class SimDF():
    """
    Simulate a dataset for binary classification of size (n x p).
    Random, normally distributed coefficients.
    Gaussian noise.
    Some irrelevant features.
    
    TODO:
    * Implement class imbalance.
    """
    def __init__(self, n, p, coef_mu, coef_sd, noise, threshold=.5, imbalance=0):
        self.n = n
        self.p = p
        self.coef_mu = coef_mu
        self.coef_sd = coef_sd
        self.noise = noise
        self.coef = []
    
    def generate(self):
        # random, normally distributed coefficients
        # some variables are irrelevant
        coef = np.append(np.random.normal(loc=self.coef_mu, scale=self.coef_sd, size=self.p//2),
                         [0 for _ in range(self.p//2)])

        # noise
        e = np.random.normal(loc=0, scale=np.sqrt(self.noise), size=self.n)
        X = np.random.normal(loc=0, scale=1, size=self.n*self.p).reshape(self.n, self.p)
        y = np.matmul(X, coef) + e
        y = 1/(1+np.exp(-y)) #logistic transformation
        y = 1*(y > .5)
        self.X = X
        self.y = y
        self.coef = coef
        return
    
    def to_dataframe(self):
        index = list(range(self.n))
        simulated_data = np.zeros((self.n, self.p+2))
        simulated_data[:, 0] = index
        simulated_data[:, 1:(self.p+1)] = self.X
        simulated_data[:, -1] = self.y 
        simulated_data = pd.DataFrame(simulated_data)
        simulated_data.columns = ['id'] + [f"X{i}" for i in range(self.p)] + ['y']
        return simulated_data
    
    def __str__(self):
        s = f"Simulated dataset of size ({self.n} x {self.p}), with noise variance {self.noise}.\nCoefficients: {self.coef}"
        return s
        