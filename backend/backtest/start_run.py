import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as Axes
import seaborn as sns
import numpy as np


def clean_numeric(col: str, col_replace: str):
    return pd.to_numeric(col.str.replace(col_replace, ""), errors="coerce")


def parse_csv(path: str) -> pd.DataFrame:
    csv_data = pd.read_csv("./rendement.csv", parse_dates=["date"])

    csv_data["rendement_predit"] = clean_numeric(csv_data["rendement_predit"], "%")
    csv_data["vol_empirique"] = clean_numeric(csv_data["vol_empirique"], "%")
    csv_data["alpha(expo_crypto)"] = clean_numeric(csv_data["alpha(expo_crypto)"], "%")
    csv_data["rendement_observe"] = clean_numeric(csv_data["rendement_observe"], "%")
    csv_data["valeur_portefeuille"] = clean_numeric(
        csv_data["valeur_poterfeuille"], "%"
    )
    csv_data["valeur_crypto"] = clean_numeric(csv_data["valeur_crypto"], "%")
    csv_data.dropna(inplace=True)
    return csv_data


def create_line_chart_axe(axe: Axes, data_x: any, data_y: any, title: str, mark: str):
    axe.plot(data_x, data_y, label=title, marker=mark)
    axe.set_title(title)
    axe.set_xlabel("Date")
    axe.set_ylabel("Rendement Prédit (%)")
    axe.tick_params(labelrotation=45)
    axe.legend()


def start_run():
    csv_data = parse_csv("./rendement.csv")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

    create_line_chart_axe(
        axes[0], csv_data["date"], csv_data["rendement_predit"], "Rendement Prédit", "o"
    )
    fig.show()
    # # Line Chart for 'rendement_observe' over time
    # plt.figure(figsize=(10, 5))
    # # Plot 'rendement_predit' over time
    # plt.plot(
    #     csv_data["date"],
    #     csv_data["rendement_predit"],
    #     label="Rendement Prédit",
    #     marker="o",
    # )

    # # Plot 'rendement_observe' over time
    # plt.plot(
    #     csv_data["date"],
    #     csv_data["rendement_observe"],
    #     label="Rendement Observé",
    #     marker="x",
    # )
    # sns.lineplot(data=csv_data, x="date", y="rendement_observe")
    # plt.title("Observed Return Over Time")
    # plt.xlabel("Date")
    # plt.ylabel("Observed Return (%)")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig("./observed_return_over_time.png")

    # # Bar Chart for the count of 'prediction' categories
    # prediction_counts = csv_data["prediction"].value_counts()
    # plt.figure(figsize=(7, 5))
    # sns.barplot(x=prediction_counts.index, y=prediction_counts.values)
    # plt.title("Prediction Category Counts")
    # plt.xlabel("Prediction Category")
    # plt.ylabel("Count")
    # plt.tight_layout()
    # plt.savefig("./prediction_category_counts.png")

    # # Pie Chart for the distribution of 'prediction' categories
    # plt.figure(figsize=(7, 7))
    # csv_data["prediction"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90)
    # plt.title("Distribution of Prediction Categories")
    # plt.ylabel("")
    # plt.tight_layout()

    # # Plotting the line chart
    # plt.figure(figsize=(12, 6))
    # plt.plot(
    #     csv_data["date"],
    #     csv_data["rendement_predit"],
    #     label="Rendement Prédit",
    #     marker="o",
    # )
    # plt.plot(
    #     csv_data["date"],
    #     csv_data["rendement_observe"],
    #     label="Rendement Observé",
    #     marker="x",
    # )
    # plt.title("Comparaison du Rendement Prédit et Observé sur le Temps")
    # plt.xlabel("Date")
    # plt.ylabel("Rendement (%)")
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.tight_layout()

    # Show the plots
    # plt.show()
