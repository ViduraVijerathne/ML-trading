import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com")
SYMBOL = os.getenv("SYMBOL", "SOLUSDT")
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", "1.0"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "10.0"))
