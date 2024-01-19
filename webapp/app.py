import logging
import logging.config

import uvicorn
from fastapi import FastAPI

from pydantic import BaseModel
from typing import Dict

api = FastAPI(title="Crypto Backtest", docs_url="/docs")


class ConfigModel(BaseModel):
    wallet_amount: float
    start_date: str
    ml_model: str
    backtest_strategy: str
    transaction_fee: float
    cryptos: Dict[str, bool]


@api.get("/")
async def root():
    return {"message": "Hello, world!"}


@api.post("/config")
async def config(body: ConfigModel):
    """Config endpoint. Sets the configuration of the backtest.

    Returns:
        OK
    """
    return {"message": "Hello, world!"}


@api.get("/run")
async def run_test():
    return {"message": "Hello, world!"}


@api.get("/cancel")
async def cancel():
    return {"message": "Hello, world!"}


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8080, log_level="info", reload=True)
