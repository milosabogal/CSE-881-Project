import torch

import numpy as np

from cvxopt import matrix 
from cvxopt import solvers

SMA_PERIOD = 20
EMA_PERIOD = 20
CCI_PERIOD = 20
VOLATILITY_PERIOD = 20
ROC_PERIOD = 20

def returns(df, ticker):
    return (df[f"Close {ticker}"] - df[f"Open {ticker}"]) / df[f"Open {ticker}"]

def log_returns(df, ticker):
    return np.log(df[f"Close {ticker}"]).diff()

def sma(df, ticker):
    return df[f"Close {ticker}"].rolling(window=SMA_PERIOD).mean()

def ema(df, ticker):
    return df[f"Close {ticker}"].ewm(span=EMA_PERIOD, adjust=False).mean()

def vwap(df, ticker):
    value = df[f"Close {ticker}"] * df[f"Volume {ticker}"]
    cumulative_value = value.cumsum()
    cumulative_volume = df[f"Volume {ticker}"].cumsum()
    return cumulative_value / cumulative_volume

def cci(df, ticker):
    typical_price = (df[f"High {ticker}"] + df[f"Low {ticker}"] + df[f"Close {ticker}"]) / 3
    mean_typical_price = typical_price.rolling(window=CCI_PERIOD).mean()
    mean_deviation = (typical_price - mean_typical_price).abs().rolling(window=CCI_PERIOD).mean()
    return (typical_price - mean_typical_price) / (0.015 * mean_deviation)

def volatility(df, ticker):
    return df[f"Returns {ticker}"].rolling(window=VOLATILITY_PERIOD).std()

def roc(df, ticker):
    return (df[f"Close {ticker}"] / df[f"Close {ticker}"].shift(ROC_PERIOD) - 1) * 100

def create_regression_dataset(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset)-window_size):
        features, target = dataset[i:i+window_size], dataset[i+window_size, -1]
        X.append(features)
        y.append(target)
    return torch.tensor(np.array(X), dtype=torch.float), torch.tensor(np.array(y), dtype=torch.float).reshape(-1,1)

def create_classification_dataset(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset)-window_size):
        features, target = dataset[i:i+window_size, :-1], dataset[i+window_size, -1]
        X.append(features)
        y.append(target)
    return torch.tensor(np.array(X), dtype=torch.float), torch.tensor(np.array(y), dtype=torch.long)

def markowitz_mean_variance(returns, covariance_matrix, risk_tolerance=0):
    """ Compute the optimal weights using the Markowitz mean-variance model.

    The goal is to maximize the expected returns of a portfolio for a given level of risk.
    Since we don't allow short selling (negative weights), we use G and h in cvxopt to indicate the inequality w_i >= 0.
    
    Args:
        returns: Returns of the tickers
        covariance_matrix: Covariance matrix of the returns of the tickers
        risk_tolerance: Risk tolerance parameter

    Returns:
        The optimal weights
    """
    solvers.options["show_progress"] = False
    P = covariance_matrix
    q = -risk_tolerance * returns
    G = -np.eye(len(returns))
    h = np.zeros((len(returns),1))
    A = np.ones((1, len(returns)))
    b = 1.0

    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    
    sol = solvers.qp(P, q, G, h, A, b)
    
    return np.array(sol["x"]).squeeze()

# Adapted from https://goldinlocks.github.io/Time-Series-Cross-Validation/
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits, train_size):
        self.n_splits = n_splits
        self.train_size = train_size
    
    def split(self, X, window_size):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = -window_size
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(self.train_size * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
