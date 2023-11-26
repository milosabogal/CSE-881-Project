import model
import numpy as np

def submitTickers(t1, t2, t3, r, i):
    investment = float(i)
    risk_tol = float(r)
    tickers = [t1, t2, t3]
    pOpt = model.PortfolioOptimizer(tickers, risk_tol)

    pOpt.get_data()
    pOpt.construct_features()
    pOpt.preprocess()
    pOpt.train_models(print_progress=True)
    pOpt.get_predictions()

    optWeights = pOpt.get_optimal_weights()
    optimal_weights_percentages = np.round(optWeights*100, 2)
    print()

    for i in range(len(tickers)):
        print(f"You should invest {round(optWeights[i] * investment, 2)} ({optimal_weights_percentages[i]}%) in {tickers[i]}.")

    return (optWeights, optimal_weights_percentages, tickers, investment)
