import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import yfinance as yf
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error

import warnings

import processing as ps

START_DATE = "2021-01-01"
END_DATE = "2024-01-01"

TRAIN_PROPORTION = 0.9

WINDOW_SIZE = 20
HIDDEN_DIM = 64
LR = 0.0005
N_EPOCHS = 300
BATCH_SIZE = 16

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                            hidden_size = self.hidden_dim, 
                            num_layers=self.num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0)) 
        out = self.linear(hn[-1]) # Last hidden state
        
        return out
    
class PortfolioOptimizer:
    def __init__(self, ticker_symbols, risk_tolerance, method=None):
        self.ticker_symbols = ticker_symbols
        
        self.start_date = START_DATE
        self.end_date = END_DATE
        
        self.risk_tolerance = risk_tolerance
        
        self.method = method

        self.feature_functions = {"Returns" : ps.returns, 
                                  "Log Returns" : ps.log_returns,
                                  "SMA" : ps.sma,
                                  "EMA" : ps.ema,
                                  "VWAP": ps.vwap,
                                  "CCI": ps.cci,
                                  "Volatility" : ps.volatility ,
                                  "RoC" : ps.roc
                                  }

    def get_data(self):
        self.dataframes = {}
        print("Collecting data...")
        for ticker_symbol in self.ticker_symbols:
            data = yf.download(ticker_symbol, start=self.start_date, end=self.end_date)
            self.dataframes[ticker_symbol] = data
        print()

    def construct_features(self, target_feature = "Returns", feature_list = None):
        if feature_list == None: # Use all features
            feature_list = list(self.feature_functions)
        
        for ticker_df in self.dataframes.values():
            for feature in feature_list:
                feature_function = self.feature_functions[feature]
                ticker_df[feature] = feature_function(ticker_df)

            ticker_df.drop(["Open", "High", "Low", "Close", "Adj Close"], axis = 1, inplace=True)
            ticker_df.dropna(inplace=True)
            target = ticker_df.pop(target_feature)
            ticker_df[target_feature] = target

    def preprocess(self):
        self.train_scalers = {}
        self.test_scalers = {}
        self.train_datasets = {}
        self.train_labels = {}
        self.test_datasets = {}
        self.test_labels = {}

        for ticker_symbol, ticker_df in self.dataframes.items():
            X = ticker_df.to_numpy()

            train_size = int(TRAIN_PROPORTION * X.shape[0])
            train, test = X[:train_size], X[train_size:]

            train_scaler = RobustScaler()
            train_dataset = train_scaler.fit_transform(train) 

            test_scaler = RobustScaler()
            test_dataset = test_scaler.fit_transform(test) 

            X_train, y_train = ps.create_dataset(train_dataset, window_size=WINDOW_SIZE)
            X_test, y_test = ps.create_dataset(test_dataset, window_size=WINDOW_SIZE)

            self.train_datasets[ticker_symbol] = X_train
            self.train_labels[ticker_symbol] = y_train
            self.test_datasets[ticker_symbol] = X_test
            self.test_labels[ticker_symbol] = y_test
            self.train_scalers[ticker_symbol] = train_scaler
            self.test_scalers[ticker_symbol] = test_scaler

    def train_models(self, method=None, print_progress=False):
        self.models = {}
        for ticker_symbol in self.ticker_symbols:
            if print_progress:
                print(f"Training for {ticker_symbol}")
            self.models[ticker_symbol] = self.train_model(ticker_symbol, print_progress)
            if print_progress:
                print()
    
    def train_model(self, ticker_symbol, print_progress):
        X_train = self.train_datasets[ticker_symbol]
        y_train = self.train_labels[ticker_symbol]
        train_scaler = self.train_scalers[ticker_symbol]

        model = LSTM(input_dim=X_train.shape[2], hidden_dim=HIDDEN_DIM, num_layers=1, output_dim=1)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        torch.manual_seed(1)

        loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)

        for epoch in range(N_EPOCHS+1):
            model.train()
            for X_batch, y_batch in loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch.reshape(-1,1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 20 != 0:
                continue
            
            if print_progress:
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_train)
                    # Repeat column to match dimension of scaler and get only last column (target) after descaling
                    y_pred_orig = train_scaler.inverse_transform(np.repeat(y_pred, X_train.shape[2], axis=-1))[:,-1] 
                    y_train_orig = train_scaler.inverse_transform(np.repeat(y_train, X_train.shape[2], axis=-1))[:,-1]
                    train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_pred_orig))
                    print("Epoch %d: train RMSE %.4f" % (epoch, train_rmse))
            
        return model
    
    def get_predictions(self):
        self.returns_predictions = {}
        for ticker_symbol in self.ticker_symbols:
            self.returns_predictions[ticker_symbol] = self.get_prediction(ticker_symbol)

    def get_prediction(self, ticker_symbol):
        model = self.models[ticker_symbol]
        test_scaler = self.test_scalers[ticker_symbol]
        
        data = self.dataframes[ticker_symbol].to_numpy()
        test_dataset = test_scaler.transform(data[-(WINDOW_SIZE+1):])
        X_test, _ = ps.create_dataset(test_dataset, WINDOW_SIZE)

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            y_pred_orig = test_scaler.inverse_transform(np.repeat(y_pred, X_test.shape[2], axis=-1))[:,-1]
            return y_pred_orig

    def get_optimal_weights(self):
        predicted_returns = np.array(list(self.returns_predictions.values()))
        historical_covariance_matrix = np.cov([ticker_df["Returns"] for ticker_df in self.dataframes.values()])
        weights = ps.markowitz_mean_variance(predicted_returns, historical_covariance_matrix, self.risk_tolerance)
        return weights
