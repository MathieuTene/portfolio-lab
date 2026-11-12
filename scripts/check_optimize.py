import pandas as pd

from portfolio_lab.optimize import optimize
from portfolio_lab.data import download_prices

prices = download_prices()

w_sharpe = optimize(prices, 'max_sharpe')
w_minvol = optimize(prices, 'min_vol')
w_hrp = optimize(prices, 'hrp')
print(pd.DataFrame({'max_sharpe': w_sharpe, 'min_vol': w_minvol, 'hrp': w_hrp}).round(3))