import pandas as pd
import numpy as np
from dataclasses import dataclass
import sys
import os

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from strategies.base import Strategy, StrategyConfig

class NaiveStrategy(Strategy):
    def __init__(self, config: StrategyConfig, dataset_path: str) -> None:
        Strategy.__init__(self, config, dataset_path)
        self.crypto_std_devs = {}

    def predict_return(self, crypto_name, current_time):
        # Si rendement hier > 0 
        return np.random.normal()
        
    def run_strategy(self, current_row, current_time, days_elapsed = 0) -> None:
        for name in self.config.cryptos:
            predicted_return = self.predict_return(name, current_time)
            tendance = 0.5 * self.crypto_std_devs[name]
            if predicted_return > tendance:
                # Allocation haussière
                self.buy(name, self.portfolio['risk_free'])
            elif predicted_return < - tendance:
                # Allocation baissière
                self.send(name, self.portfolio[name] * 0.5)


if __name__ == "__main__":
    # Exemple d'utilisation de la classe Strategy
    cryptos: dict[str, bool] = {
        "BTC": True,
        "ETH": True,
        "DOGE": True,
        "XRP": False,
        "DOT": False,
        "BCH": False,
        "SOL": False,
        "ADA": False,
        "MATIC": False,
        "BNB": False,
        "LTC": False,
    }
    config = StrategyConfig(start_date='2018-08-17', transaction_fee=0.0, wallet_amount=10000.0, cryptos=cryptos)
    strategy = NaiveStrategy(config, './webapp/strategies/test.xlsx')
    strategy.load_data()
    strategy.run_pipeline()
    portfolio_status = strategy.get_status()
    results = strategy.get_results()
    print(results)
    #results.to_csv('strategy_output.csv', index=False)