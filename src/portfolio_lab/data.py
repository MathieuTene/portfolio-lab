from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import START, TICKERS

CACHE = Path('data/prices.parquet')

def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    prices = prices.dropna(how='all')
    prices = prices.ffill(limit=5)
    return prices.dropna()

def download_prices(tickers=TICKERS, start=START, end=None, use_cache=True) -> pd.DataFrame:
    tickers = list(tickers)
    if use_cache and CACHE.exists():
        cached = pd.read_parquet(CACHE)
        if not cached.empty and set(tickers).issubset(cached.columns):
            return cached[tickers]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = clean_prices(raw['Close'][tickers])
    if prices.empty:
        raise ValueError('Aucune donnée téléchargée : vérifie START et les tickers.')
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(CACHE)
    return prices