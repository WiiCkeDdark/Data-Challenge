"""Librairie qui contient les formules pour remplir un fichier backtest.csv
"""
import pandas as pd



def predict_category(rendement_predit, vol_empirique):
    if rendement_predit > 0.5 * vol_empirique:
        return 'bull'
    elif rendement_predit < -0.5 * vol_empirique:
        return 'bear'
    else:
        return 'neutral'

def calculate_alpha(prediction):
    if prediction == 'bull':
        return 1.0
    elif prediction == 'bear':
        return 0.5
    else:
        return 1.0  # La logique exacte n'est pas claire pour 'neutral'

def calculate_portfolio_value(previous_value, alpha, rendement_observe):
    return previous_value * (1 + alpha * rendement_observe)

def calculate_crypto_value(portfolio_value, alpha):
    return portfolio_value * alpha