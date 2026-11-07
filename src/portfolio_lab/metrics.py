import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def annualized_return(returns):
    n = len(returns)
    total = (1 + returns).prod()
    return total ** (TRADING_DAYS / n) - 1


def annualized_vol(returns):
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, rf=0.0):
    return (annualized_return(returns) - rf) / annualized_vol(returns)


def sortino_ratio(returns, rf=0.0):
    downside = np.sqrt((np.minimum(returns, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    return (annualized_return(returns) - rf) / downside


def drawdown_series(returns):
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns):
    return drawdown_series(returns).min()


def historical_var(returns, level=0.95):
    return -returns.quantile(1 - level)


def historical_cvar(returns, level=0.95):
    var = returns.quantile(1 - level)
    return -returns[returns >= var].mean()


def summary(returns: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    out = pd.DataFrame({
        'Rendement annualisé': annualized_return(returns),
        'Volatilité annualisée': annualized_vol(returns),
        'Sharpe': sharpe_ratio(returns, rf),
        'Sortino': sortino_ratio(returns, rf),
        'Drawdown max': max_drawdown(returns),
        'VaR 95% (jour)': historical_var(returns),
        'CVaR 95% (jour)': historical_cvar(returns)
    })
    out['Calmar'] = out['Rendement annualisé'] / out['Drawdown max'].abs()
    return out