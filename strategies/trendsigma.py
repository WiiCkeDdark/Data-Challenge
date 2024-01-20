import pandas as pd
import numpy as np
from dataclasses import dataclass
import sys
import os
from strategies.trend import TrendStrategy
import xgboost as xgb
from sklearn.impute import SimpleImputer

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from strategies.base import Strategy, StrategyConfig

BULL = 1
BEAR = 2
NEUTRAL = 3

class TrendSigmaStrategy(TrendStrategy):
    def __init__(self, config: StrategyConfig, dataset_path: str, output_path: str) -> None:
        TrendStrategy.__init__(self, config, dataset_path, output_path)
        self.models: dict[str, any] = {
            "BTC": self.load_xgb_model("./models/BTC_sigma.model"),
            "ETH": self.load_xgb_model("./models/ETH.model"),
            "XRP": self.load_xgb_model("./models/XRP.model"),
            "LTC": self.load_xgb_model("./models/LTC.model"),
            "SOL": self.load_xgb_model("./models/SOL.model")
        }
        self.models: dict[str, any] = {
            "BTC": self.load_xgb_model("./models/BTC_sigma.model"),
        }
    