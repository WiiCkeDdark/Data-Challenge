from dataclasses import dataclass
import os
import pandas as pd
from pandas import Series, DataFrame
import xgboost as xgb

data_backtest_csv = [
    [
        "date",
        "rendement_predit",
        "vol_empirique",
        "prediction",
        "alpha(expo_crypto)",
        "rendement_observe",
        "valeur_portefeuille",
        "valeur_crypto",
    ]
]

@dataclass
class StrategyConfig:
    start_date: str
    transaction_fee: float
    wallet_amount: float
    cryptos: dict[str, bool]


class Strategy:
    def __init__(self, config: StrategyConfig, dataset_path: str, output_path: str) -> None:
        self.config = config
        self.dataset_path = dataset_path
        self.dataset = None
        self.config = config
        self.portfolio = {'total': 0, 'risk_free': config.wallet_amount, 'global_rendement': 0}
        self.portfolio_sum = 0
        self.root_folder = output_path
        self.output_folder = "outputs"
        self.portfolio_history = []
        self.test_datasets: dict[str, DataFrame] = {}
        self.backtests_outputs: dict[str, list[list]] = {}
        self.res_output = []
        self.size: int = 0
        self.headers = []
        
        print("Voici les cryptomonnaies prises en compte dans cette simulation :")
        for name in config.cryptos:
            if self.config.cryptos[name] is True:
                print(name)
                self.portfolio[name] = 0
    
    def preprocess(self, crypto_name: str) -> pd.DataFrame:
        return None
        
    def export_backtest_csv(self) -> pd.DataFrame:
        print("Génération des fichiers backtest au format .csv")
        os.makedirs(self.root_folder, exist_ok=True)
        for i in self.config.cryptos:
            if self.config.cryptos[i] is True:
                new_df = pd.DataFrame(self.backtests_outputs[i][1:], columns=self.backtests_outputs[i][0])
                new_df.to_csv(os.path.join(self.root_folder, f"{i}_backtest.csv"), index=False)
    
    def export_output_csv(self) -> pd.DataFrame:
        print("Génération des fichiers output au format .csv")
        os.makedirs(self.root_folder, exist_ok=True)
        print(self.headers)
        new_df = pd.DataFrame(self.res_output, columns=self.headers)
        new_df.to_csv(os.path.join(self.root_folder, f"output.csv"), index=False)
    
    def fill_observation(self, crypto_name, current_row, previous_row):
        if self.compute_rendement(crypto_name, current_row, previous_row) > 0.5 * self.crypto_std_devs[crypto_name]:
            return "bull"
        elif self.compute_rendement(crypto_name, current_row, previous_row) > - 0.5 * self.crypto_std_devs[crypto_name]:
            return "bear"
        else:
            return "neutral"
    
    def load_xgb_model(self, path: str) -> any:
        # Charge le modèle de machine learning pour une crypto-monnaie spécifique
        print(path)
        model = xgb.Booster()
        model.load_model(path)
        return model
    
    def compute_vol_empirique(self):
        """Calcule l'écart-type des rendements quotidiens des cryptos ou volatilité empirique
        """
        print("Calcul de l'écart type des rendements quotidiens de chaque crypto")
        for name in self.config.cryptos:
            crypto_returns = self.dataset[f'Close_{name}'].pct_change().dropna()
            self.crypto_std_devs[name] = crypto_returns.std()
            print(name, ": ", self.crypto_std_devs[name])
    
    def run_pipeline(self) -> None:
        self.headers.append("date")
        for name in self.config.cryptos:
            if self.config.cryptos[name] is True:
                self.test_datasets[name] = self.preprocess(name)
                self.backtests_outputs[name] = data_backtest_csv.copy()
                self.headers.append(f"prevision_{name}")
                self.headers.append(f"observation_{name}")
        self.compute_vol_empirique()
        self.size = len(self.dataset)
        days_elapsed = 0
        print("Déroulement de la stratégie...")
        for row in self.dataset.iterrows():
            self.run_strategy(self.dataset.iloc[days_elapsed], self.test_datasets, days_elapsed)
            if days_elapsed > 0:
                self.compute_portofolio_sum(self.dataset.iloc[days_elapsed], self.dataset.iloc[days_elapsed - 1])
                self.portfolio_history.append(self.portfolio.copy())
            days_elapsed += 1
            self.debug()
        self.export_backtest_csv()
        #self.export_output_csv()
    
    def run_strategy() -> None:
        pass
    
    def compute_base_100(self, crypto_name: str, current_row, days_elapsed: int, previous_base_100):
        return (self.compute_rendement(crypto_name, current_row, self.dataset.iloc[days_elapsed - 1]) + 1) * previous_base_100
    
    def compute_rendement(self, crypto_name: str, current_row: Series, previous_row: Series) -> float:
        return (current_row[f'Close_{crypto_name}'] - previous_row[f'Close_{crypto_name}']) / previous_row[f'Close_{crypto_name}']
    
    def compute_portofolio_sum(self, current_row: Series, previous_row: Series):
        total: float = 0
        for i in self.config.cryptos:
            if self.config.cryptos[i] is True:
                total += self.portfolio[i] * current_row[f'Close_{i}']
        self.portfolio['total'] = self.portfolio['risk_free'] + total
    
    def buy(self, current_row: pd.Series, crypto_name: str, amount: float, fees: float = 0):
        if amount == 0:
            return
        print("J'achète", crypto_name, "pour ", amount)
        self.portfolio['risk_free'] -= amount
        self.portfolio[crypto_name] += amount / current_row[f'Close_{crypto_name}'] # 
            
    def send(self, current_row: pd.Series, crypto_name: str, amount: float, fees: float = 0):
        if amount == 0:
            return
        print("Je vends", crypto_name, "pour ", amount)
        self.portfolio[crypto_name] -= amount
        self.portfolio['risk_free'] += amount * current_row[f'Close_{crypto_name}'] * (1 - fees)
    
    def get_portofolio_history(self):
        # Retourne l'historique du portefeuille
        return pd.DataFrame(self.portfolio_history, columns=['Time', 'Portfolio'])
    
    def debug(self):
        print("Portofolio: ", self.portfolio)
        #print("Portofolio total =", self.portfolio_history)
    
    def load_data(self):
        """Charger les données depuis le fichier CSV/Excel
        """
        try:
            print("Chargement du dataset de test")
            self.dataset = pd.read_excel(self.dataset_path, index_col=0)
            start_test_date = self.config.start_date
            end_test_date = self.dataset.index[-1]
            print("Start test date: ", start_test_date)
            print("End test date: ", end_test_date)
            self.dataset = self.dataset.loc[start_test_date:end_test_date]

        except Exception:
            print("Le dataset de test n'a pas pu être chargé")
            exit(84)
        print("Le dataset de test a été chargé correctement")