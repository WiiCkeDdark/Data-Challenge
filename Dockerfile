FROM python:3.10-slim-buster as base

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend .

COPY crypto_backtest .

EXPOSE 8050

CMD ["python", "backtest/dashboard.py", "./"]