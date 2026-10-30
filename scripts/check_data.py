from portfolio_lab.data import download_prices

prices = download_prices()
print(prices.shape, prices.index.min(), prices.index.max())
print(prices.isna().sum()) 
print(prices.tail())

returns = prices.pct_change().dropna()
print(returns.abs().max())