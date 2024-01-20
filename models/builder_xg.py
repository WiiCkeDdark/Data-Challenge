import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

df = pd.read_excel(
    "federalfinancegestion/Rennes_DataChallenge2024_Cryptomarkets_dataset.xlsx",
    index_col=0,
)
start_train_date = df.index[0]
end_train_date = "31-08-2022"
start_test_date = "01-09-2022"
end_test_date = df.index[-1]

crypto = "DOGE"

print("Start train date: ", start_train_date)
print("End train date: ", end_train_date)
print("Start test date: ", start_test_date)
print("End test date: ", end_test_date)
# convert df index to datetime
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
df_train = df.loc[start_train_date:end_train_date]
df_test = df.loc[start_test_date:end_test_date]
# print number of nan
print("Number of nan in train: ", df_train.isna().sum().sum())
print("Number of nan in test: ", df_test.isna().sum().sum())

x_train = df_train.drop(columns=["Target"])
y_train = df_train[["Target"]]
x_test = df_test.drop(columns=["Target"])
y_test = df_test[["Target"]]

# @#####################################################################################################################``
# convert to DMatrix
d_train = xgb.DMatrix(x_train, label=y_train)
d_test = xgb.DMatrix(x_test, label=y_test)


# Définir les paramètres à optimiser
param_grid = {
    "max_depth": [i for i in range(2, 20, 1)],
    "eta": [0.2, 0.3, 0.4, 0.5],
    "objective": ["multi:softmax"],
    "num_class": [3],
}
# Définir le nombre maximum d'epochs
max_epochs = 1000

# Définir le nombre d'epochs pour early stopping
early_stopping_rounds = 10

# Créer un classificateur XGBoost avec early_stopping_rounds
xgb_model = xgb.XGBClassifier(early_stopping_rounds=early_stopping_rounds)
# Créer un objet GridSearchCV
grid_search = GridSearchCV(
    xgb_model, param_grid, cv=3, verbose=1, n_jobs=-1, scoring="f1_weighted"
)

# Ajuster GridSearchCV au jeu de données d'entraînement
grid_search.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

# Obtenir les meilleurs paramètres
best_params = grid_search.best_params_
print(best_params)

# Entraîner le modèle avec les meilleurs paramètres
model = xgb.train(best_params, d_train, max_epochs)

# Prédire
preds = model.predict(d_test)

# Évaluer
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy}")

# print unique values of preds
print(np.unique(preds))

# get the confusion matrix
cm = confusion_matrix(y_test, preds)

# plot it
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{crypto}_confusion_matrix.png")

# Training the model with best parameters
best_model = xgb.XGBClassifier(**best_params)
best_model.fit(x_train, y_train)

# Extracting feature importance
feature_importances = best_model.feature_importances_

# Assuming you have feature names
feature_names = x_train.columns

# Pair feature names with their importance scores
features = sorted(
    zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True
)

# Show features and their importance
for feature, importance in features:
    print(f"Feature: {feature}, Importance: {importance}")

# Créer un graphique de l'importance des variables
fig, ax = plt.subplots(figsize=(30, 30))  # Augmenter la taille du graphique
xgb.plot_importance(model, ax=ax, height=0.5)  # Ajuster la hauteur de chaque barre

# Rotation des étiquettes si nécessaire
plt.xticks(rotation=45)

# save
plt.savefig(f"{crypto}.png")

# save model in file
model.save_model(f"{crypto}.json")
