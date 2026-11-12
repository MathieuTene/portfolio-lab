import pandas as pd
from pypfopt import EfficientFrontier, HRPOpt, expected_returns, risk_models
import scipy.cluster.hierarchy as sch

# PyPortfolioOpt lit cet attribut privé, absent des SciPy récents
if not hasattr(sch, '_LINKAGE_METHODS'):
    sch._LINKAGE_METHODS = {
        'single', 'complete', 'average', 'centroid', 'median', 'ward', 'weighted'
    }


def optimize(prices: pd.DataFrame, method: str = 'max_sharpe', rf: float = 0.02, max_weight: float = 0.30) -> pd.Series:
    if method == 'hrp':
        returns = prices.pct_change().dropna()
        weights = HRPOpt(returns).optimize()
        return pd.Series(weights).reindex(prices.columns).fillna(0.0)

    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, max_weight))
    if method == 'max_sharpe':
        try:
            ef.max_sharpe(risk_free_rate=rf)
        except ValueError:
            # aucun actif ne bat le taux sans risque sur la fenêtre : repli prudent
            ef = EfficientFrontier(mu, cov, weight_bounds=(0, max_weight))
            ef.min_volatility()
    elif method == 'min_vol':
        ef.min_volatility()
    else:
        raise ValueError(f'méthode inconnue : {method}')
    return pd.Series(ef.clean_weights()).reindex(prices.columns).fillna(0.0)