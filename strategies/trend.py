import pandas as pd
import numpy as np
from dataclasses import dataclass
import sys
import os
import xgboost as xgb
from sklearn.impute import SimpleImputer

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from strategies.base import Strategy, StrategyConfig

BULL = 1
BEAR = 2
NEUTRAL = 3

class TrendStrategy(Strategy):
    def __init__(self, config: StrategyConfig, dataset_path: str, output_path: str) -> None:
        Strategy.__init__(self, config, dataset_path, output_path)
        self.crypto_std_devs = {}
        print("Chargement des modèles...")
        self.models: dict[str, any] = {
            "BTC": self.load_xgb_model("./models/btc.json"),
            "ETH": self.load_xgb_model("./models/ETH.json"),
            "DOGE": self.load_xgb_model("./models/DOGE.json")
        }
        print("Modèles chargés")

    def predict_return(self, current_row: pd.Series, crypto_name: str):
        df = pd.DataFrame(current_row).transpose()
        df = xgb.DMatrix(df)
        prediction = self.models[crypto_name].predict(df)
        return prediction
        
    def run_strategy(self, row: pd.Series, test_datasets: dict[str, pd.DataFrame], days_elapsed: int) -> None:
        rendement: str = "0%"
        prediction: str = "neutral"
        alpha: str = "100%"
    
        for name in self.config.cryptos:
            if self.config.cryptos[name] is True:
                predicted_return = self.predict_return(test_datasets[name].iloc[days_elapsed], name)
                if predicted_return == 1:            
                    self.buy(row, name, self.portfolio['risk_free'])
                    rendement = "2%"
                    prediction = "bull"
                    alpha = "100%"
                elif predicted_return == 2:
                    self.send(row, name, self.portfolio[name] * 0.5)
                    rendement = "-2%"
                    prediction = "bear"
                    alpha = "50%"
                self.backtests_outputs[name].append(
                    [
                        row.iloc[0],
                        rendement,
                        "4%",
                        prediction,
                        alpha,
                        self.compute_rendement(name, row, self.dataset.iloc[days_elapsed - 1]),
                        self.portfolio[name] * row[f"Close_{name}"],
                        row[f"Close_{name}"]
                    ]
                )

    def preprocess(self, crypto_name: str) -> pd.DataFrame:
        df = self.dataset.copy()
        start_test_date = self.config.start_date
        print("Start test date: ", start_test_date)
        # convert df index to datetime
        df.index = pd.to_datetime(df.index)

        columns_to_keep = [col for col in df.columns if crypto_name.upper() in col]
        # add btc lowercase
        columns_to_keep += [col for col in df.columns if crypto_name.lower() in col]
        columns_to_keep += [
            "uncertainty_attention_twitter",
            "cbdc_attention_twitter",
            "regulation_attention_twitter",
            "VStoxx",
            "Gold",
            "ECRPUS1YIndex",
            "WTI_CrudeOil",
            "Brent_CrudeOil",
            "PercentHikeCut",
        ]
        # add all column start with "monp"
        columns_to_keep += [col for col in df.columns if "monp" in col]
        columns_to_keep += [col for col in df.columns if "sdcc" in col]
        columns_to_keep += [col for col in df.columns if "crisis" in col]
        columns_to_keep += [col for col in df.columns if "_Index" in col]
        if crypto_name.upper() != "BTC":
            columns_to_keep += [col for col in df.columns if "btc" in col]
            columns_to_keep += [col for col in df.columns if "BTC" in col]
        df = df[columns_to_keep]
        # Initialize the SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        # Select btc_ columns
        columns_to_computes = [col for col in df.columns if crypto_name.upper() in col]
        columns_to_computes += [col for col in df.columns if crypto_name.lower() in col]
        columns_to_computes += [col for col in df.columns if "monp_" in col]
        columns_to_computes += [col for col in df.columns if "Gold" in col]
        columns_to_computes += [col for col in df.columns if "VStoxx" in col]
        columns_to_computes += [col for col in df.columns if "WTI_CrudeOil" in col]
        columns_to_computes += [col for col in df.columns if "Brent_CrudeOil" in col]
        columns_to_computes += [col for col in df.columns if "_Index" in col]
        if crypto_name.upper() != "BTC":
            columns_to_computes += [col for col in df.columns if "btc" in col]
            columns_to_computes += [col for col in df.columns if "BTC" in col]
        # Fit and transform the btc_ columns
        imputed_data = imp.fit_transform(df[columns_to_computes])
        df.loc[:, columns_to_computes] = imputed_data
        # convert Close_BTC to rendement
        df[f"Close_{crypto_name.upper()}"] = df[f"Close_{crypto_name.upper()}"].pct_change()
        # shift df["Close_BTC"] to have the rendement of the next day
        df[f"Close_{crypto_name.upper()}"] = df[f"Close_{crypto_name.upper()}"].shift(-1)
        # if rendement > 0 then 1 else 0 and if rendement < 0 then -1 else 0
        df["Target"] = 0
        df.loc[df[f"Close_{crypto_name.upper()}"] > 0.02, "Target"] = 1
        df.loc[df[f"Close_{crypto_name.upper()}"] < -0.02, "Target"] = 2
        # drop Close_BTC
        df = df.drop(columns=[f"Close_{crypto_name.upper()}"])
        # ! df = df.drop(df.index[-1])
        df_test = df.loc[start_test_date:]
        # print number of nan
        print("Number of nan in test: ", df_test.isna().sum().sum())
        x_test = df_test.drop(columns=["Target"])
        return x_test

if __name__ == "__main__":
    # Exemple d'utilisation de la classe Strategy
    config = StrategyConfig(start_date='2022-09-01', transaction_fee=0.0, wallet_amount=10000.0, cryptos=cryptos)
    strategy = TrendStrategy(config, './strategies/test.xlsx', "./backup_examples")
    strategy.load_data()
    strategy.run_pipeline()
    results = strategy.get_results()
    print(results)
    #results.to_csv('strategy_output.csv', index=False)