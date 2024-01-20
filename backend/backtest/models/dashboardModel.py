from graphs.create_graphs import create_scatter_line, create_pie_chart, create_figure
from utils.clean_csv import clean_numeric
from utils.find_crypto import find_crypto_df
from utils.load_csv import load_csv
import pandas as pd
from dash import dcc

crypto_objects = load_csv()


class Dashboard:
    csv_data = find_crypto_df(crypto_objects, "BTC")

    def __init__(self, data: pd.DataFrame):
        self.csv_data = data

    def create_wallet_chart(self):
        valeur_portefeuille = create_scatter_line(
            x_data=self.csv_data["date"],
            y_data=self.csv_data["valeur_portefeuille"],
            trace_name="Valeur portefeuille",
        )

        valuer_crypto = create_scatter_line(
            x_data=self.csv_data["date"],
            y_data=self.csv_data["valeur_crypto"],
            trace_name="Valeur Crypto",
        )

        return dcc.Graph(
            id="valeur_portefeuille",
            figure=create_figure(
                [valeur_portefeuille, valuer_crypto],
                "Comparaison de la valeur du portefeuille et valeur de la crypto",
                ["Date", "Valeur en $"],
            ),
        )

    def create_rend_graph(self):
        # Clean and convert percentage columns to floats

        self.csv_data["rendement_predit"] = pd.to_numeric(
            self.csv_data["rendement_predit"].astype(str).str.replace("%", ""),
            errors="coerce",
        )
        self.csv_data["rendement_observe"] = clean_numeric(
            self.csv_data["rendement_observe"], "%"
        )
        # Create Plotly graph objects for both rendement_predit and rendement_observe
        rendement_predit_trace = create_scatter_line(
            x_data=self.csv_data["date"],
            y_data=self.csv_data["rendement_predit"],
            trace_name="Rendement Prédit",
        )

        rendement_observe_trace = create_scatter_line(
            x_data=self.csv_data["date"],
            y_data=self.csv_data["rendement_observe"],
            trace_name="Rendement Observé",
        )
        return dcc.Graph(
            id="rendement-predit-vs-observe",
            figure=create_figure(
                [rendement_predit_trace, rendement_observe_trace],
                "Comparaison du Rendement Prédit et Observé sur le Temps",
                ["Date", "Rendement (%)"],
            ),
        )

    def create_prediction_pie(self):
        # Pie chart data
        prediction_labels = self.csv_data["prediction"].value_counts().index
        prediction_values = self.csv_data["prediction"].value_counts().values

        # Create a pie chart
        prediction_pie_chart = create_pie_chart(
            labels=prediction_labels,
            values=prediction_values,
            chart_name="Distribution des catégories de prédictions",
        )
        return dcc.Graph(id="prediction-distribution", figure=prediction_pie_chart)
