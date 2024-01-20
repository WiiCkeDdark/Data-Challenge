from strategies.naive import NaiveStrategy
from strategies.trend import TrendStrategy
from strategies.trendsigma import TrendSigmaStrategy
import typer
from typing import List, Optional
from strategies.base import Strategy, StrategyConfig 

app = typer.Typer()


strategies: dict[str, Strategy] = {
    "xgboost": TrendStrategy,
    "xgboost-sigma": TrendSigmaStrategy,
    "naive": NaiveStrategy,
}

cryptos: dict[str, bool] = {
            "BTC": True,
            "ETH": True,
            "DOGE": True,
            "XRP": False,
            "DOT": False,
            "BCH": False,
            "SOL": False,
            "ADA": False,
            "MATIC": False,
            "BNB": False,
            "LTC": False,
        }

@app.command()
def run(all: bool = False, 
         crypto: Optional[str] = typer.Option(None, "--crypto"),
         output: Optional[str] = typer.Option("./backtest_examples", "--output"),
         strategy: Optional[str] = typer.Option("xgboost", "--strategy"),
         dataset: Optional[str] = typer.Option("federalfinancegestion/Rennes_DataChallenge2024_Cryptomarkets_dataset.xlsx", "--dataset"),
         start_date: Optional[str] = typer.Option('2022-09-01', "--start-date"),
         fee: Optional[float] = typer.Option(0, "--fee"),
         start_wallet: Optional[float] = typer.Option(10000, "--wallet"),
         debug: bool = False
) :
    if all:
        typer.echo("Testing all cryptocurrencies")
    elif crypto:
        typer.echo(f"Testing selected cryptocurrencies: {list}")
        for i in cryptos:
            cryptos[i] = False
        cryptos[crypto] = True
                
    config = StrategyConfig(start_date=start_date, transaction_fee=fee, wallet_amount=start_wallet, cryptos=cryptos)
    
    try:
        strategy = strategies[strategy](config, dataset, output_path=output)
        strategy.load_data()
        strategy.run_pipeline()
    except KeyError:
        print("Cette stratégie n'éxiste pas")


@app.command()
def visualize(
    list: bool = typer.Option(False, "--list"),
    output: Optional[str] = typer.Option(None, "--output"),
    ):
    if list:
        typer.echo(" for visualization")
        # Ajoutez ici la logique pour lister les cryptomonnaies disponibles

if __name__ == "__main__":
    app()
