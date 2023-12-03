import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import yfinance as yf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from joblib import Parallel, delayed

import processing as ps

from itertools import product

START_DATE = "2021-01-01"
END_DATE = "2023-12-01"
TRAIN_SIZE = 0.9
NUM_FOLDS = 5

RANDOM_ITERATIONS = 50

WINDOW_SIZE_VALUES = [3, 5, 10]
BATCH_SIZE_VALUES = [32, 64]
HIDDEN_SIZE_VALUES = [32, 64, 128]
LEARNING_RATE_VALUES = [0.1, 0.01, 0.001, 0.0001]
EPOCH_VALUES = [100, 400, 700]


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[-1])  # Last hidden state

        return out


class PortfolioOptimizer:
    DEFAULT_DAILY_INVESTMENT_AMOUNT = 100

    def __init__(self, ticker_symbols, device, start_date=START_DATE, end_date=END_DATE, train_size=TRAIN_SIZE):
        self.ticker_symbols = ticker_symbols
        self.risk_tolerance = None
        self.method = None
        self.start_date = start_date
        self.end_date = end_date
        self.train_size = train_size
        self.device = device

        self.num_testing_days = 0

        self.feature_functions = {"Returns": ps.returns,
                                  "Log Returns": ps.log_returns,
                                  "SMA": ps.sma,
                                  "EMA": ps.ema,
                                  "VWAP": ps.vwap,
                                  "CCI": ps.cci,
                                  "Volatility": ps.volatility,
                                  "RoC": ps.roc
                                  }
        self.dataframes = {}

        self.historical_covariance_matrix = None

        self.best_regression_models = {}
        self.regression_predictions = {}

        self.best_classification_model = None
        self.classification_predictions = None
        self.greedy_classification = False

    def get_data(self):
        for ticker_symbol in self.ticker_symbols:
            data = yf.download(ticker_symbol, start=self.start_date, end=self.end_date)
            self.dataframes[ticker_symbol] = data

    def print_data_summary(self):
        example_df = self.dataframes[self.ticker_symbols[0]]
        total_num_days = example_df.shape[0]
        num_features = example_df.shape[1] - 1
        print(f"Number of samples: {total_num_days}")
        print(f"Number of features: {num_features}")

    def construct_features(self, target_feature="Returns"):
        print("\nBefore preprocessing:")
        self.print_data_summary()
        for ticker, ticker_df in self.dataframes.items():

            ticker_df.rename(columns={'Volume': f"Volume {ticker}",
                                      'Open': f"Open {ticker}",
                                      'High': f"High {ticker}",
                                      'Low': f"Low {ticker}",
                                      'Close': f"Close {ticker}",
                                      'Adj Close': f"Adj Close {ticker}"}, inplace=True)

            for feature, feature_function in self.feature_functions.items():
                ticker_df[f"{feature} {ticker}"] = feature_function(ticker_df, ticker)

            ticker_df.dropna(inplace=True)
            target = ticker_df.pop(f"{target_feature} {ticker}")
            ticker_df[f"{target_feature} {ticker}"] = target

        self.historical_covariance_matrix = np.cov(
            [ticker_df[f"Returns {ticker}"].iloc[:int(self.train_size * len(ticker_df))]
             for ticker, ticker_df in self.dataframes.items()])

        print("\nAfter preprocessing:")
        self.print_data_summary()
        example_df = self.dataframes[self.ticker_symbols[0]]
        total_num_days = example_df.shape[0]
        self.num_testing_days = total_num_days - int(self.train_size * total_num_days)
        print(f"Number of testing days: {self.num_testing_days}")

    def set_method(self, method):
        self.method = method

    def set_risk_tolerance(self, risk_tolerance):
        self.risk_tolerance = risk_tolerance

    def set_greedy_classification(self, greedy_classification):
        self.greedy_classification = greedy_classification

    def parallel_cross_validate(self, full_data, num_folds=NUM_FOLDS, ticker_symbol=None):

        def evaluate_params(params):
            window_size, batch_size, hidden_dim, learning_rate, num_epochs = params
            total_error = 0
            btscv = ps.BlockingTimeSeriesSplit(n_splits=num_folds, train_size=TRAIN_SIZE)

            for train_indices, test_indices in btscv.split(full_data, window_size):
                train, test = full_data[train_indices], full_data[test_indices]
                total_error += self.train_and_test(train, test, window_size, batch_size,
                                                   hidden_dim, learning_rate, num_epochs, ticker_symbol)[0]

            average_error = total_error / num_folds
            return average_error, (window_size, batch_size, hidden_dim, learning_rate, num_epochs)

        params_list = list(product(WINDOW_SIZE_VALUES, BATCH_SIZE_VALUES, HIDDEN_SIZE_VALUES,
                                   LEARNING_RATE_VALUES, EPOCH_VALUES))

        results = Parallel(n_jobs=-1, verbose=6)(delayed(evaluate_params)(params) for params in params_list)

        best_error, best_params = min(results, key=lambda x: x[0])
        print("\nBest parameters: ws = {}, batch size = {}, hid = {}, lr = {}, epochs = {}".format(*best_params))
        print("Lowest average error:", best_error, "\n")
        return best_params

    def train_and_test(self, train, test, window_size, batch_size, hidden_dim, learning_rate, num_epochs,
                       ticker_symbol=None, print_progress=False, display_metrics=False, final_predictions=False):
        if self.method == "regression":
            output_dim = 1
            loss_string = "RMSE"
            loss_fn = nn.MSELoss()

            scaler = RobustScaler()
            train_scaled = scaler.fit_transform(train)
            test_scaled = scaler.transform(test)

            X_train, y_train = ps.create_regression_dataset(train_scaled, window_size=window_size)
            X_test, y_test = ps.create_regression_dataset(test_scaled, window_size=window_size)
            X_train, X_test = X_train.to(self.device), X_test.to(self.device)
            y_train, y_test = y_train.to(self.device), y_test.to(self.device)

        else:
            output_dim = len(self.ticker_symbols)
            loss_string = "Cross Entropy Loss"
            loss_fn = nn.CrossEntropyLoss()

            train_features, train_labels = train[:, :-1], train[:, -1].reshape(-1, 1)
            test_features, test_labels = test[:, :-1], test[:, -1].reshape(-1, 1)

            # Add one-hot encoding of labels from previous days as features
            one_hot_train_labels = np.eye(len(self.ticker_symbols))[train_labels.astype(int).reshape(-1)]
            one_hot_test_labels = np.eye(len(self.ticker_symbols))[test_labels.astype(int).reshape(-1)]

            scaler = RobustScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)

            train_dataset = np.concatenate([train_features_scaled, one_hot_train_labels, train_labels], axis=1)
            test_dataset = np.concatenate([test_features_scaled, one_hot_test_labels, test_labels], axis=1)

            X_train, y_train = ps.create_classification_dataset(train_dataset, window_size=window_size)
            X_test, y_test = ps.create_classification_dataset(test_dataset, window_size=window_size)
            X_train, X_test, y_train, y_test = X_train.to(self.device), X_test.to(self.device), y_train.to(
                self.device), y_test.to(self.device)

        model = LSTM(input_dim=X_train.shape[-1], hidden_dim=hidden_dim, num_layers=1, output_dim=output_dim,
                     device=self.device).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)

        # Train
        losses = []
        epoch_list = []
        for epoch in range(num_epochs + 1):
            model.train()
            for X_batch, y_batch in loader:
                y_pred_logits = model(X_batch)
                loss = loss_fn(y_pred_logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 20 != 0:
                continue

            if print_progress or display_metrics:
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_train)
                    if self.method == "regression":
                        # Repeat column to match dimension of scaler and get only last column (target) after descaling
                        y_pred_orig = scaler.inverse_transform(
                            np.repeat(y_pred.cpu().numpy(), X_train.shape[2], axis=-1))[:, -1]
                        y_train_orig = scaler.inverse_transform(
                            np.repeat(y_train.cpu().numpy(), X_train.shape[2], axis=-1))[:, -1]
                        train_error = np.sqrt(mean_squared_error(y_train_orig, y_pred_orig))

                    elif self.method == "classification":
                        train_error = loss_fn(y_pred, y_train).item()

                    if print_progress:
                        print(f"Epoch {epoch}: train {loss_string} = {train_error:.4f}")
                    if display_metrics:
                        losses.append(train_error)
                        epoch_list.append(epoch)

        if display_metrics:
            plt.figure()
            plt.plot(epoch_list, losses, '-o')
            plt.xlabel("Epoch")
            plt.ylabel(loss_string)
            if self.method == "regression":
                title = f"{ticker_symbol} Training {loss_string} vs Epoch"
            else:
                title = f"Training {loss_string} vs Epoch "
            plt.title(title)

        # Test
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            if self.method == "regression":
                # Repeat column to match dimension of scaler and get only last column (target) after descaling
                y_pred_orig = scaler.inverse_transform(np.repeat(y_pred.cpu().numpy(), X_test.shape[2], axis=-1))[:, -1]
                y_test_orig = scaler.inverse_transform(np.repeat(y_test.cpu().numpy(), X_test.shape[2], axis=-1))[:, -1]
                test_error = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

                if display_metrics:
                    plt.figure(figsize=(16, 4))
                    plt.plot(y_test_orig, 'o-k', label="actual")
                    plt.plot(y_pred_orig, 'o-r', label="predicted")
                    plt.xlabel("Time (days)")
                    plt.ylabel("Returns")
                    plt.legend()
                    plt.title(f'{ticker_symbol} Test RMSE = ' + str(test_error))

                if final_predictions:
                    self.regression_predictions[ticker_symbol] = y_pred_orig

            elif self.method == "classification":
                test_error = loss_fn(y_pred, y_test).item()
                if display_metrics:
                    y_pred_labels = torch.argmax(y_pred, dim=1)
                    accuracy = accuracy_score(y_test.cpu().numpy(), y_pred_labels.cpu().numpy())
                    print("Test Cross Entropy Loss:", test_error)
                    print("Testing accuracy:", accuracy)
                    print(confusion_matrix(y_test.cpu().numpy(), y_pred_labels.cpu().numpy()))
                    print(classification_report(y_test.cpu().numpy(), y_pred_labels.cpu().numpy()))

                if final_predictions:
                    pred_probabilities = torch.softmax(y_pred, dim=1).cpu().numpy()
                    self.classification_predictions = pred_probabilities

            return test_error, model

    def train_regression(self):
        for ticker_symbol in self.ticker_symbols:
            ticker_df = self.dataframes[ticker_symbol]
            X = ticker_df.to_numpy()
            train_size = int(self.train_size * X.shape[0])
            X_validation = X[:train_size]

            print(f"\nCross validating for {ticker_symbol}...")
            best_params = self.parallel_cross_validate(X_validation, ticker_symbol=ticker_symbol)
            window_size, batch_size, hidden_dim, learning_rate, num_epochs = best_params
            train, test = X[:train_size], X[train_size - window_size:]
            self.best_regression_models[ticker_symbol] = self.train_and_test(train, test, window_size,
                                                                             batch_size, hidden_dim, learning_rate,
                                                                             num_epochs, ticker_symbol,
                                                                             print_progress=False, display_metrics=True,
                                                                             final_predictions=True)[1]

    def train_classification(self):
        full_df = pd.concat(self.dataframes.values(), axis=1)
        full_df["Label"] = np.argmax(full_df[[f"Returns {ticker}" for ticker in self.ticker_symbols]].values, axis=1)
        X = full_df.to_numpy()
        train_size = int(self.train_size * X.shape[0])
        X_validation = X[:train_size]

        print("\nCross validating for classification...")
        window_size, batch_size, hidden_dim, learning_rate, num_epochs = self.parallel_cross_validate(X_validation)
        train, test = X[:train_size], X[train_size - window_size:]
        self.best_classification_model = self.train_and_test(train, test, window_size, batch_size, hidden_dim,
                                                             learning_rate, num_epochs, print_progress=False,
                                                             display_metrics=True, final_predictions=True)[1]

    def train(self):
        torch.manual_seed(0)
        np.random.seed(0)

        if self.method == "equal" or self.method == "random":
            return

        elif self.method == "regression":
            self.train_regression()

        elif self.method == "classification":
            self.train_classification()

        else:
            raise ValueError("Invalid method")

    def get_weights(self, day=-1):
        if self.method == "equal":
            return np.ones(len(self.ticker_symbols)) / len(self.ticker_symbols)

        elif self.method == "regression":
            predicted_returns = np.array([self.regression_predictions[ticker][day] for ticker in self.ticker_symbols])
            weights = ps.markowitz_mean_variance(predicted_returns, self.historical_covariance_matrix,
                                                 self.risk_tolerance)
            return weights

        elif self.method == "classification":
            weights = self.classification_predictions[day]
            if self.greedy_classification:
                weights = np.eye(len(self.ticker_symbols))[np.argmax(weights)]
            return weights

        elif self.method == "random":
            random_numbers = np.random.rand(len(self.ticker_symbols))
            return random_numbers / np.sum(random_numbers)

        else:
            raise ValueError("Invalid method")

    def get_portfolio_returns(self):
        portfolio_returns_list = np.zeros(self.num_testing_days)
        for i in range(-self.num_testing_days + 1, 0):
            true_returns = []
            for ticker, ticker_df in self.dataframes.items():
                ticker_returns = ticker_df[f"Returns {ticker}"].iloc[i]
                true_returns.append(ticker_returns)

            true_returns = np.array(true_returns)
            weights = self.get_weights(day=i)
            current_portfolio_returns = np.dot(weights, true_returns)
            portfolio_returns_list[i] = current_portfolio_returns

        return portfolio_returns_list

    def plot_portfolio_performance(self):
        if self.method == "random":
            portfolio_returns_list = np.zeros(self.num_testing_days)
            for _ in range(RANDOM_ITERATIONS):
                portfolio_returns_list += self.get_portfolio_returns()
            portfolio_returns_list = portfolio_returns_list / RANDOM_ITERATIONS

        else:
            portfolio_returns_list = self.get_portfolio_returns()

        self._plot_results(portfolio_returns_list)

    def _plot_results(self, portfolio_returns_list):
        plt.figure(figsize=(16, 4))
        if self.method == "regression":
            plt.suptitle(f"LSTM Regression + Markowitz Mean-Variance Model (Risk Tolerance = {self.risk_tolerance})")

        elif self.method == "classification":
            if self.greedy_classification:
                classification_title = "LSTM Classification Model (Greedy)"
            else:
                classification_title = "LSTM Classification Model"
            plt.suptitle(classification_title)

        elif self.method == "random":
            plt.suptitle("Average Performance for Random Portfolio Allocation")

        elif self.method == "equal":
            plt.suptitle("Equal Portfolio Allocation")

        ax1 = plt.subplot(1, 2, 1)
        portfolio_return_percentages = 100 * portfolio_returns_list
        mu = portfolio_return_percentages.mean()
        sigma = portfolio_return_percentages.std(ddof=1)
        text_str = '\n'.join((
            r'$\mu=%.2f$' % (mu,),
            r'$\sigma=%.2f$' % (sigma,)))

        ax1.plot(portfolio_return_percentages)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax1.set_ylabel('Daily Portfolio Returns (in %)')
        ax1.set_xlabel('Day')
        ax1.set_title('Daily Portfolio Returns')

        ax2 = plt.subplot(1, 2, 2)
        cumulative_portfolio_returns = np.cumprod(1 + np.array(portfolio_returns_list)) - 1
        ax2.plot(100 * cumulative_portfolio_returns)
        ax2.set_ylabel('Cumulative Portfolio Returns (in %)')
        secondary_axis = ax2.secondary_yaxis('right',
                                             functions=(lambda x: self.DEFAULT_DAILY_INVESTMENT_AMOUNT * (x / 100 + 1),
                                                        lambda x: 100 * (x / self.DEFAULT_DAILY_INVESTMENT_AMOUNT - 1)))
        secondary_axis.set_ylabel('Capital (in USD)')
        plt.xlabel('Day')
        plt.title("Cumulative Portfolio Performance")

        plt.tight_layout()
