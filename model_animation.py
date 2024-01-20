import xgboost as xgb
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation


crypto = "ETH"


def load_xgb_model(path: str) -> any:
    model = xgb.Booster()
    model.load_model(path)
    return model


df = pd.read_excel(
    "federalfinancegestion/Rennes_DataChallenge2024_Cryptomarkets_dataset.xlsx",
    index_col=0,
)

dataset = pd.read_excel(
    "federalfinancegestion/Rennes_DataChallenge2024_Cryptomarkets_dataset.xlsx",
    index_col=0,
)

index = df.index
df.index = pd.to_datetime(df.index)
# keep only row with name containe BTC
columns_to_keep = [col for col in df.columns if crypto.upper() in col]
# add btc lowercase
columns_to_keep += [col for col in df.columns if crypto.lower() in col]
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
if crypto.upper() != "BTC":
    columns_to_keep += [col for col in df.columns if "btc" in col]
    columns_to_keep += [col for col in df.columns if "BTC" in col]
df = df[columns_to_keep]
# Initialize the SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
# Select btc_ columns
columns_to_computes = [col for col in df.columns if crypto.upper() in col]
columns_to_computes += [col for col in df.columns if crypto.lower() in col]
columns_to_computes += [col for col in df.columns if "monp_" in col]
columns_to_computes += [col for col in df.columns if "Gold" in col]
columns_to_computes += [col for col in df.columns if "VStoxx" in col]
columns_to_computes += [col for col in df.columns if "WTI_CrudeOil" in col]
columns_to_computes += [col for col in df.columns if "Brent_CrudeOil" in col]
columns_to_computes += [col for col in df.columns if "_Index" in col]
if crypto.upper() != "BTC":
    columns_to_computes += [col for col in df.columns if "btc" in col]
    columns_to_computes += [col for col in df.columns if "BTC" in col]
# Fit and transform the btc_ columns
imputed_data = imp.fit_transform(df[columns_to_computes])
df.loc[:, columns_to_computes] = imputed_data
# convert Close_BTC to rendement
df[f"Close_{crypto.upper()}"] = df[f"Close_{crypto.upper()}"].pct_change()
# shift df["Close_BTC"] to have the rendement of the next day
df[f"Close_{crypto.upper()}"] = df[f"Close_{crypto.upper()}"].shift(-1)
# if rendement > 0 then 1 else 0 and if rendement < 0 then -1 else 0
df["Target"] = 0
df.loc[df[f"Close_{crypto.upper()}"] > 0.02, "Target"] = 1
df.loc[df[f"Close_{crypto.upper()}"] < -0.02, "Target"] = 2
# drop Close_BTC
df = df.drop(columns=[f"Close_{crypto.upper()}"])
# drop last row
df = df.drop(df.index[-1])

start_date = "2019-01-01"
end_date = "2020-01-01"
df_test = df.loc[start_date:end_date]
x_test = df_test.drop(columns=["Target"])

mdl = load_xgb_model("./btc.model")
crypto_holdings = 0
usd_balance = 10000
wallet_values = []
stophold = []
dates = []
crypto_prices = []
idx = 0

head_btc = dataset.iloc[0]
head_price = head_btc["Close_BTC"]
stophold_crypto = usd_balance / head_price


def compute_rendement(
    crypto_name: str, current_price: float, previous_price: float, current_value: float
) -> float:
    tmp = current_price / previous_price - 1
    current_value += current_value * tmp
    return current_value


def predict_category(rendement_predit, vol_empirique):
    if rendement_predit > 0.5 * vol_empirique:
        return "bull"
    elif rendement_predit < -0.5 * vol_empirique:
        return "bear"
    else:
        return "neutral"


def calculate_alpha(prediction):
    if prediction == 1:
        return 1.0
    elif prediction == "bear":
        return 0.5
    else:
        return 1.0  # La logique exacte n'est pas claire pour 'neutral'


def calculate_portfolio_value(previous_value, alpha, rendement_observe):
    return previous_value * (1 + alpha * rendement_observe)


def calculate_crypto_value(portfolio_value, alpha):
    return portfolio_value * alpha


def compute_vol_empirique():
    return "4%"


col = [
    "date",
    "rendement_predit",
    "vol_empirique",
    "prediction",
    "alpha(expo_crypto)",
    "rendement_observe",
    "valeur_portefeuille",
    "valeur_crypto",
]
data = [[]]

for row in x_test.iterrows():
    df = pd.DataFrame(row[1]).transpose()
    current_series = dataset.iloc[idx]
    current_price = current_series["Close_BTC"]

    stophold_value = stophold_crypto * current_price
    stophold.append(stophold_value)
    gMatrix = xgb.DMatrix(df)
    idx += 1
    prediction = "neutral"
    rendement = "0%"
    preds_daily = mdl.predict(gMatrix)

    if preds_daily[0] == 1 and usd_balance != 0:
        crypto_bought = usd_balance / current_price
        crypto_holdings += crypto_bought
        rendement = "2%"
        prediction = "bull"
        usd_balance = 0
        print("Buy on", row[0], " - Bought", crypto_bought, "crypto")
    elif preds_daily[0] == 2 and crypto_holdings != 0:
        usd_balance += crypto_holdings * current_price
        rendement = "-2%"
        prediction = "bear"
        print("Sell on", row[0], " - Sold", crypto_holdings, "crypto")
        crypto_holdings = 0

    total_wallet_value = usd_balance + (crypto_holdings * current_price)
    wallet_values.append(total_wallet_value)
    crypto_prices.append(current_price)
    dates.append(row[0])
    data.append([row[0], rendement, "4%", prediction])

new_df = pd.DataFrame(data[1:], columns=col)


data = pd.DataFrame(
    {
        "Date": dates * 2,
        "Value": wallet_values + stophold,
        "Metric": ["Wallet Value"] * len(wallet_values)
        + ["StopHold Price"] * len(stophold),
    }
)
plt.figure(figsize=(12, 6))
sns.lineplot(x="Date", y="Value", hue="Metric", style="Metric", data=data)
plt.xlabel("Date")
plt.ylabel("Wallet Value (USD)")
plt.title("Evolution of Wallet Value and StopHold Price Over Time")

plt.show()
