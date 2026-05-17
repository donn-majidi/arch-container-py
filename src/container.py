# assets: AAPL, NVDA
# models: ARCH, GARCH, EGARCH, GJR-GARCH
# distributions: normal, student-t

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from arch import arch_model


ASSETS = ["AAPL", "NVDA"]

DISTRIBUTIONS = {
    "normal": "normal",
    "student_t": "t",
}

MODELS = {
    "ARCH(1)": {"vol": "ARCH", "p": 1, "o": 0, "q": 0},

    "GARCH(1,1)": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},

    "GJR-GARCH(1,1)": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},

    "TGARCH(1,1)": {
        "vol": "GARCH",
        "p": 1,
        "o": 1,
        "q": 1,
        "power": 1.0,
    },

    "EGARCH(1,1)": {"vol": "EGARCH", "p": 1, "o": 0, "q": 1},

    "EGARCH(1,1,1)": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},

    "APARCH(1,1)": {"vol": "APARCH", "p": 1, "o": 1, "q": 1},

    "FIGARCH(1,1)": {"vol": "FIGARCH", "p": 1, "o": 0, "q": 1},
}

@dataclass
class ModelResult:
    asset: str
    model: str
    distribution: str
    aic: float
    bic: float
    loglikelihood: float
    params: Dict


def load_sp500_prices_from_excel(
    file_path: str,
    sheet_name: str = "S&P500",
    header_row: int = 6,
    asset_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)

    df.columns = df.columns.astype(str).str.strip()

    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if asset_columns is None:
        asset_columns = {
            "AAPL": "AAPL",
            "NVDA": "NVDA",
        }

    missing_cols = [
        excel_col for excel_col in asset_columns.values()
        if excel_col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"These columns were not found in the Excel file: {missing_cols}. "
            f"Available columns include: {list(df.columns[:20])}"
        )

    prices = df[list(asset_columns.values())].copy()
    prices = prices.rename(columns={v: k for k, v in asset_columns.items()})

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes percentage log returns:
        r_t = 100 * log(P_t / P_{t-1})
    """
    prices = prices.apply(pd.to_numeric, errors="coerce")
    returns = 100 * np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")

    return returns

class ArchModelContainer:
    def __init__(self, returns: pd.DataFrame):
        """
        returns: DataFrame with columns AAPL and NVDA.
        Returns should be percentage log returns.
        """
        self.returns = returns
        self.fitted_models = {}
        self.summary_table: Optional[pd.DataFrame] = None

    def fit_one_model(self, asset: str, model_name: str, distribution_name: str):
        y = pd.to_numeric(self.returns[asset], errors="coerce").dropna()

        if y.empty:
            raise ValueError(f"No valid observations available for {asset}.")

        model_config = MODELS[model_name]
        dist = DISTRIBUTIONS[distribution_name]

        # Dynamically construct ARCH-family model specification
        model_kwargs = {
            "mean": "Constant",
            "vol": model_config["vol"],
            "p": model_config["p"],
            "o": model_config["o"],
            "q": model_config["q"],
            "dist": dist,
            "rescale": False,    
        }

        if "power" in model_config:
            model_kwargs["power"] = model_config["power"]

        model = arch_model(y, **model_kwargs)        

        result = model.fit(disp="off")

        self.fitted_models[(asset, model_name, distribution_name)] = result

        return ModelResult(
            asset=asset,
            model=model_name,
            distribution=distribution_name,
            aic=result.aic,
            bic=result.bic,
            loglikelihood=result.loglikelihood,
            params=result.params.to_dict(),
        )

    def fit_all(self, assets: Optional[List[str]] = None) -> pd.DataFrame:
        assets = assets or ASSETS
        results = []

        for asset in assets:
            if asset not in self.returns.columns:
                raise ValueError(f"{asset} is not in the returns DataFrame.")

            for model_name in MODELS:
                for distribution_name in DISTRIBUTIONS:
                    result = self.fit_one_model(
                        asset=asset,
                        model_name=model_name,
                        distribution_name=distribution_name,
                    )
                    results.append(result.__dict__)

        self.summary_table = pd.DataFrame(results)
        return self.summary_table

    def get_model(self, asset: str, model_name: str, distribution_name: str):
        key = (asset, model_name, distribution_name)

        if key not in self.fitted_models:
            raise ValueError("This model has not been fitted yet.")

        return self.fitted_models[key]
