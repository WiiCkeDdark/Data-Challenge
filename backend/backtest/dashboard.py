import dash
from dash import dcc, html, Output, Input
import pandas as pd
from graphs.create_graphs import create_scatter_line, create_pie_chart, create_figure
from utils.clean_csv import clean_numeric
from datetime import date
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

# Read the CSV data into a DataFrame
csv_data = pd.read_csv("./rendement.csv", parse_dates=["date"])

# Clean and convert percentage columns to floats
csv_data["rendement_predit"] = clean_numeric(csv_data["rendement_predit"], "%")
csv_data["rendement_observe"] = clean_numeric(csv_data["rendement_observe"], "%")

# Pie chart data
prediction_labels = csv_data["prediction"].value_counts().index
prediction_values = csv_data["prediction"].value_counts().values

# Create Plotly graph objects for both rendement_predit and rendement_observe
rendement_predit_trace = create_scatter_line(
    x_data=csv_data["date"],
    y_data=csv_data["rendement_predit"],
    trace_name="Rendement Prédit",
)

rendement_observe_trace = create_scatter_line(
    x_data=csv_data["date"],
    y_data=csv_data["rendement_observe"],
    trace_name="Rendement Observé",
)

valeur_portefeuille = create_scatter_line(
    x_data=csv_data["date"],
    y_data=csv_data["valeur_portefeuille"],
    trace_name="Valeur portefeuille",
)

valuer_crypto = create_scatter_line(
    x_data=csv_data["date"],
    y_data=csv_data["valeur_crypto"],
    trace_name="Valeur Crypto",
)

# Create a pie chart
prediction_pie_chart = create_pie_chart(
    labels=prediction_labels,
    values=prediction_values,
    chart_name="Distribution des catégories de prédictions",
)

start_date = date(2017, 8, 17)
end_date = date(2023, 3, 24)


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
        dcc.DatePickerRange(
            id="date-picker-range",
            min_date_allowed=start_date,
            max_date_allowed=end_date,
            initial_visible_month=start_date,
            start_date=start_date,
            end_date=end_date,
        ),
        html.Div(
            id="value-card",
            className="card",
            children=[
                html.Div(
                    className="card-content",
                    children=[
                        html.Span("Valeur Portefeuille:", className="card-title"),
                        html.H3(id="value-indicator", className="indicator"),
                    ],
                ),
                html.Div(
                    id="growth-percentage",
                    className="growth-info",
                    children=[
                        html.Span("Growth Percentage:", className="growth-title"),
                        html.H3(id="growth-value", className="growth-value"),
                    ],
                ),
            ],
        ),
        dcc.Graph(id="prediction-distribution", figure=prediction_pie_chart),
        dcc.Graph(
            id="rendement-predit-vs-observe",
            figure=create_figure(
                [rendement_predit_trace, rendement_observe_trace],
                "Comparaison du Rendement Prédit et Observé sur le Temps",
                ["Date", "Rendement (%)"],
            ),
        ),
        dcc.Graph(
            id="valeur_portefeuille",
            figure=create_figure(
                [valeur_portefeuille, valuer_crypto],
                "Comparaison de la valeur du portefeuille et valeur de la crypto",
                ["Date", "Valeur en $"],
            ),
        ),
        html.Div(id="output-container-date-picker-range"),
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
    # Filter the DataFrame based on the date range
    filtered_df = csv_data[
        (csv_data["date"] >= start_date) & (csv_data["date"] <= end_date)
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
    filtered_df = csv_data[
        (csv_data["date"] >= start_date) & (csv_data["date"] <= end_date)
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
        end_date = csv_data["date"].max()

    # Convert end_date to datetime
    end_date = pd.to_datetime(end_date) if end_date else csv_data["date"].max()

    # Find the row in the dataframe where the date is the end_date
    closest_end = csv_data[csv_data["date"] <= end_date]["date"].max()
    value_at_end_date = csv_data.loc[
        csv_data["date"] == closest_end, "valeur_portefeuille"
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
    start_date = pd.to_datetime(start_date) if start_date else csv_data["date"].min()
    end_date = pd.to_datetime(end_date) if end_date else csv_data["date"].max()

    # Get the closest available dates in the dataset
    closest_start = csv_data["date"].min()
    closest_end = csv_data[csv_data["date"] <= end_date]["date"].max()

    # Retrieve values at closest start and end dates
    start_value = csv_data.loc[
        csv_data["date"] == closest_start, "valeur_portefeuille"
    ].iloc[0]
    end_value = csv_data.loc[
        csv_data["date"] == closest_end, "valeur_portefeuille"
    ].iloc[0]

    # Calculate growth percentage
    if pd.notna(start_value) and pd.notna(end_value) and start_value != 0:
        growth_percentage = ((end_value - start_value) / start_value) * 100
        color = "green" if growth_percentage >= 0 else "red"
        return html.Span(f"{growth_percentage:.2f}%", style={"color": color})
    else:
        return "Data not available"


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
