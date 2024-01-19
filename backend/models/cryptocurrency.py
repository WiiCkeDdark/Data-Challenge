import json


class Cryptocurrency:
    def __init__(self, id, symbol, name, start_date):
        self.id = id
        self.symbol = symbol
        self.name = name
        self.start_date = start_date

    def __str__(self):
        return (
            f"Cryptocurrency({self.id}, {self.symbol}, {self.name}, {self.start_date})"
        )


class CryptoConfigParser:
    def __init__(self, filepath):
        self.filepath = filepath

    def parse_config(self):
        with open(self.filepath, "r") as file:
            data = json.load(file)
            return [Cryptocurrency(**crypto) for crypto in data["cryptocurrencies"]]
