import dash
from dash import dcc, html, Output, Input
import pandas as pd
from graphs.create_graphs import create_scatter_line, create_pie_chart, create_figure
from utils.clean_csv import clean_numeric
from datetime import date
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from utils.load_csv import load_csv
from utils.find_crypto import find_crypto_df
from datetime import datetime

unique_cryptos = [
    "BTC",
    "ETH",
    "DOGE",
    "LTC",
    "ADA",
]

crypto_objects = load_csv()
date_format = "%Y-%m-%d %H:%M:%S"


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

        print(type(self.csv_data["rendement_predit"]))
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


start_date = date(2017, 8, 17)
end_date = date(2023, 3, 24)

dashboard = Dashboard(find_crypto_df(crypto_objects, unique_cryptos[0]))
predict_graph = dashboard.create_rend_graph()
wallet_chart = dashboard.create_wallet_chart()
prediction_pie = dashboard.create_prediction_pie()


# Set up the Dash app
app = dash.Dash(__name__)


# Define the layout of the app
app.layout = dbc.Container(
    children=[
        html.H1(
            children="Dashboard: Backtest",
            className="mb-2",
            style={"text-align": "center"},
        ),
        html.Div(
            children=[
                dcc.DatePickerRange(
                    id="date-picker-range",
                    min_date_allowed=start_date,
                    max_date_allowed=end_date,
                    initial_visible_month=start_date,
                    start_date=start_date,
                    end_date=end_date,
                    style={"marginRight": "30px"},
                ),
                dcc.Dropdown(
                    id="crypto-dropdown",
                    options=[
                        {"label": crypto, "value": crypto} for crypto in unique_cryptos
                    ],
                    value=unique_cryptos[0],
                    style={"width": "200px"},
                ),
            ],
            style={"display": "flex", "alignItems": "center"},
        ),
        html.Div(
            id="crypto-content",
            children=[
                html.Div(
                    id="value-card",
                    className="card",
                    children=[
                        html.Div(
                            className="card-content",
                            children=[
                                html.Span(
                                    "Valeur Portefeuille:", className="card-title"
                                ),
                                html.H3(id="value-indicator", className="indicator"),
                            ],
                        ),
                        html.Div(
                            id="growth-percentage",
                            className="growth-info",
                            children=[
                                html.Span(
                                    "Growth Percentage:", className="growth-title"
                                ),
                                html.H3(id="growth-value", className="growth-value"),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="chart-content",
                    children=[
                        html.Div(
                            id="all_charts",
                            children=[
                                prediction_pie,
                                predict_graph,
                                wallet_chart,
                            ],
                        )
                    ],
                ),
                html.Div(id="output-container-date-picker-range"),
            ],
        ),
    ]
)


@app.callback(
    Output("rendement-predit-vs-observe", "figure"),
    [
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
    ],
)
def update_rendement_graph(start_date, end_date):
    dashboard_csv_data_as_datetime = pd.to_datetime(
        dashboard.csv_data["date"], format=date_format
    )

    # Filter the DataFrame based on the date range
    filtered_df = dashboard.csv_data[
        (dashboard_csv_data_as_datetime >= start_date)
        & (dashboard_csv_data_as_datetime <= end_date)
    ]
    rendement_predit_trace = create_scatter_line(
        x_data=filtered_df["date"],
        y_data=filtered_df["rendement_predit"],
        trace_name="Rendement Prédit",
    )

    rendement_observe_trace = create_scatter_line(
        x_data=filtered_df["date"],
        y_data=filtered_df["rendement_observe"],
        trace_name="Rendement Observé",
    )

    return create_figure(
        [rendement_observe_trace, rendement_predit_trace],
        "Comparaison du Rendement Prédit et Observé sur le Temps",
        ["Date", "Rendement (%)"],
    )


@app.callback(
    Output("valeur_portefeuille", "figure"),
    [
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
    ],
)
def update_portefeuille_graph(start_date, end_date):
    dashboard_csv_data_as_datetime = pd.to_datetime(
        dashboard.csv_data["date"], format=date_format
    )
    filtered_df = dashboard.csv_data[
        (dashboard_csv_data_as_datetime >= start_date)
        & (dashboard_csv_data_as_datetime <= end_date)
    ]

    valeur_portefeuille = create_scatter_line(
        x_data=filtered_df["date"],
        y_data=filtered_df["valeur_portefeuille"],
        trace_name="Valeur portefeuille",
    )

    valuer_crypto = create_scatter_line(
        x_data=filtered_df["date"],
        y_data=filtered_df["valeur_crypto"],
        trace_name="Valeur Crypto",
    )

    return create_figure(
        [valeur_portefeuille, valuer_crypto],
        "Comparaison de la valeur du portefeuille et valeur de la crypto",
        ["Date", "Valeur en $"],
    )


@app.callback(
    Output("value-indicator", "children"), [Input("date-picker-range", "end_date")]
)
def update_value_card(end_date):
    # Handle the case where end_date is None
    if end_date is None:
        end_date = dashboard.csv_data["date"].max()

    # Convert end_date to datetime
    end_date = (
        pd.to_datetime(end_date) if end_date else dashboard.csv_data["date"].max()
    )

    # Find the row in the dataframe where the date is the end_date
    dashboard_csv_data_as_datetime = pd.to_datetime(
        dashboard.csv_data["date"], format=date_format
    )
    closest_end = dashboard.csv_data[dashboard_csv_data_as_datetime <= end_date][
        "date"
    ].max()
    value_at_end_date = dashboard.csv_data.loc[
        dashboard_csv_data_as_datetime == closest_end, "valeur_portefeuille"
    ].iloc[0]
    # If the value is not found, it means end_date is not in the dataframe, return a default message
    if pd.isna(value_at_end_date):
        return "No data for selected date"

    # Return the value as a children of the 'value-indicator' component
    return f"{value_at_end_date:,.2f}"  # Format the number with comma as thousand separator and 2 decimal places


@app.callback(
    Output("growth-value", "children"),
    [Input("date-picker-range", "start_date"), Input("date-picker-range", "end_date")],
)
def update_growth_percentage(start_date, end_date):
    # Convert dates to datetime and handle None values
    start_date = (
        pd.to_datetime(start_date) if start_date else dashboard.csv_data["date"].min()
    )
    end_date = (
        pd.to_datetime(end_date) if end_date else dashboard.csv_data["date"].max()
    )

    # Get the closest available dates in the dataset
    closest_start = dashboard.csv_data["date"].min()
    dashboard_csv_data_as_datetime = pd.to_datetime(
        dashboard.csv_data["date"], format=date_format
    )
    closest_end = dashboard.csv_data[dashboard_csv_data_as_datetime <= end_date][
        "date"
    ].max()

    # Retrieve values at closest start and end dates
    start_value = dashboard.csv_data.loc[
        dashboard.csv_data["date"] == closest_start, "valeur_portefeuille"
    ].iloc[0]
    end_value = dashboard.csv_data.loc[
        dashboard.csv_data["date"] == closest_end, "valeur_portefeuille"
    ].iloc[0]

    # Calculate growth percentage
    if pd.notna(start_value) and pd.notna(end_value) and start_value != 0:
        growth_percentage = ((end_value - start_value) / start_value) * 100
        color = "green" if growth_percentage >= 0 else "red"
        return html.Span(f"{growth_percentage:.2f}%", style={"color": color})
    else:
        return "Data not available"


from dash.dependencies import Input, Output


@app.callback(Output("chart-content", "children"), [Input("crypto-dropdown", "value")])
def update_output(selected_crypto):
    dashboard.csv_data = find_crypto_df(crypto_objects, selected_crypto)
    return html.Div(
        id="all_charts",
        children=[
            dashboard.create_prediction_pie(),
            dashboard.create_rend_graph(),
            dashboard.create_wallet_chart(),
        ],
    )


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
