import torch

import numpy as np

from cvxopt import matrix 
from cvxopt import solvers

SMA_PERIOD = 20
EMA_PERIOD = 20
CCI_PERIOD = 20
VOLATILITY_PERIOD = 20
ROC_PERIOD = 20

def returns(df):
    return (df["Close"] - df["Open"]) / df["Open"]

def log_returns(df):
    return np.log(df["Close"]).diff()

def sma(df):
    return df["Close"].rolling(window=SMA_PERIOD).mean()

def ema(df):
    return df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()

def vwap(df):
    value = df["Close"] * df["Volume"]
    cumulative_value = value.cumsum()
    cumulative_volume = df["Volume"].cumsum()
    return cumulative_value / cumulative_volume

def cci(df):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    mean_typical_price = typical_price.rolling(window=CCI_PERIOD).mean()
    mean_deviation = (typical_price - mean_typical_price).abs().rolling(window=CCI_PERIOD).mean()
    return (typical_price - mean_typical_price) / (0.015 * mean_deviation)

def volatility(df):
    return df["Returns"].rolling(window=VOLATILITY_PERIOD).std()

def roc(df):
    return (df["Close"] / df["Close"].shift(ROC_PERIOD) - 1) * 100


def create_dataset(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset)-window_size):
        features, target = dataset[i:i+window_size], dataset[i+window_size, -1]
        X.append(features)
        y.append(target)
    return torch.tensor(np.array(X)).float(), torch.tensor(np.array(y)).float().reshape(-1,1)

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
