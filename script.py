import numpy as np

from model import PortfolioOptimizer

START_DATE = "2021-01-01"
END_DATE = "2024-01-01"

def main():    
    investment_amount = float(input("Enter an investment amount: ")) # Input
    print()
    ticker_symbols = []
    while True:
        symbol = input("Enter a ticker symbol (Q to stop): ").upper() # Input
        if symbol == "Q":
            break
        ticker_symbols.append(symbol)
    print()
    
    risk_tolerance = float(input("Enter a risk tolerance: ")) # Input
    print()

    portfolio_optimizer = PortfolioOptimizer(ticker_symbols, risk_tolerance)

    portfolio_optimizer.get_data()
    portfolio_optimizer.construct_features()
    portfolio_optimizer.preprocess()
    portfolio_optimizer.train_models(print_progress=True)
    portfolio_optimizer.get_predictions()
    
    optimal_weights = portfolio_optimizer.get_optimal_weights() # Output
    optimal_weights_percentages = np.round(optimal_weights*100, 2)
    print()

    for i in range(len(ticker_symbols)):
        print(f"You should invest {round(optimal_weights[i] * investment_amount, 2)} ({optimal_weights_percentages[i]}%) in {ticker_symbols[i]}.")

if __name__ == "__main__":
    main()
