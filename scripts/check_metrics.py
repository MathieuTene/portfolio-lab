from portfolio_lab.data import download_prices
from portfolio_lab.metrics import daily_returns, summary

rets = daily_returns(download_prices())
print(summary(rets, rf=0.02).round(3))