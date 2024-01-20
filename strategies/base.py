from dataclasses import dataclass
import pandas as pd
from pandas import Series, DataFrame
from strategies.output import calculate_alpha, calculate_crypto_value, calculate_portfolio_value, predict_category
import xgboost as xgb

@dataclass
class StrategyConfig:
    start_date: str
    transaction_fee: float
    wallet_amount: float
    cryptos: dict[str, bool]


class Strategy:
    def __init__(self, config: StrategyConfig, dataset_path: str) -> None:
        self.config = config
        self.dataset_path = dataset_path
        self.dataset = None
        self.config = config
        self.portfolio = {'total': 0, 'risk_free': config.wallet_amount}
        self.portfolio_sum = 0
        self.portfolio_history = []
        self.test_datasets: dict[str, DataFrame] = {}
        print("Voici les cryptomonnaies prises en compte dans cette simulation :")
        for name in config.cryptos:
            if self.config.cryptos[name] is True:
                print(name)
                self.portfolio[name] = 0
    
    def preprocess(self, crypto_name: str) -> pd.DataFrame:
        return None
    
    def fill_backtest_csv(self, df: DataFrame, path: str):
        # Ajouter les colonnes nécessaires avec des valeurs par défaut ou calculées
        df['prediction'] = df.apply(lambda row: predict_category(row['rendement_predit'], row['vol_empirique']), axis=1)
        df['alpha'] = df['prediction'].apply(calculate_alpha)
        df['valeur_portefeuille'] = self.portfolio["risk_free"]
        df['valeur_crypto'] = 0

        # Calculer les valeurs du portefeuille et de la crypto-monnaie pour chaque jour
        for i in range(1, len(df)):
            previous_value = df.at[i - 1, 'valeur_portefeuille']
            alpha = df.at[i, 'alpha']
            rendement_observe = df.at[i, 'rendement_observe'] / 100  # Convertir en pourcentage
            df.at[i, 'valeur_portefeuille'] = calculate_portfolio_value(previous_value, alpha, rendement_observe)
            df.at[i, 'valeur_crypto'] = calculate_crypto_value(df.at[i, 'valeur_portefeuille'], alpha)

        # Écrire les résultats dans un nouveau fichier CSV
        df.to_csv(path, index=False)

    def load_xgb_model(self, path: str) -> any:
        # Charge le modèle de machine learning pour une crypto-monnaie spécifique
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
        for name in self.config.cryptos:
            if self.config.cryptos[name] is True:
                self.test_datasets[name] = self.preprocess(name)
        self.compute_vol_empirique()
        days_elapsed = 0
        print("Déroulement de la stratégie...")
        for row in self.dataset.iterrows():
            self.run_strategy(self.dataset.iloc[days_elapsed], self.test_datasets, days_elapsed)
            if days_elapsed > 0:
                self.compute_portofolio_sum(self.dataset.iloc[days_elapsed], self.dataset.iloc[days_elapsed - 1])
                self.portfolio_history.append((days_elapsed, self.portfolio.copy()))
            days_elapsed += 1
            self.debug()
    
    def run_strategy() -> None:
        pass
    
    def get_results(self) -> None:
        pass
    
    def compute_rendement(self, crypto_name: str, current_row: Series, previous_row: Series) -> float:
        tmp = (current_row[f'Close_{crypto_name}'] / previous_row[f'Close_{crypto_name}'] - 1)
        print("Rendement ", crypto_name, "", (current_row[f'Close_{crypto_name}'] / previous_row[f'Close_{crypto_name}'] - 1) * 100)
        self.portfolio[crypto_name] += self.portfolio[crypto_name] * tmp
        return self.portfolio[crypto_name]
    
    def compute_portofolio_sum(self, current_row: Series, previous_row: Series):
        total: float = 0
        for i in self.config.cryptos:
            if self.config.cryptos[i] is True:
                total += self.compute_rendement(i, current_row, previous_row)
        print("Total =", total) 
        self.portfolio['total'] = self.portfolio['risk_free'] - total
    
    def buy(self, current_row: pd.Series, crypto_name: str, amount: float, fees: float = 0):
        if amount < 0:
            return
        print("J'achète", crypto_name, "pour ", amount)
        self.portfolio['risk_free'] -= amount
        self.portfolio[crypto_name] += amount - amount * fees / current_row[f'Close_{crypto_name}']
            
    def send(self, current_row: pd.Series, crypto_name: str, amount: float, fees: float = 0):
        if amount < 0:
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
        print("Le dataset de test a été chargé correctement")