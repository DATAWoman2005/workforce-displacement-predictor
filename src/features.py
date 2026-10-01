"""Feature preparation for the Pod Nova workforce transition project.

This module contains only transformations supported directly by the supplied
dataset. Categorical encoding and numeric scaling are fitted inside a
scikit-learn Pipeline so preprocessing learns only from training data.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Automation_Risk"
RISK_ORDER = ["Low", "Medium", "High"]


# Ordered fields for which the supplied categories have a clear direction.
ORDINAL_MAPS = {
    "AI_Adoption_Level": {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    },
    "Job_Growth_Projection": {
        "Decline": 0,
        "Stable": 1,
        "Growth": 2,
    },
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with simple, target-free dataset-derived features."""

    out = df.copy()

    out["ai_adoption_ord"] = out["AI_Adoption_Level"].map(
        ORDINAL_MAPS["AI_Adoption_Level"]
    )

    out["growth_ord"] = out["Job_Growth_Projection"].map(
        ORDINAL_MAPS["Job_Growth_Projection"]
    )

    out["remote_flag"] = out["Remote_Friendly"].eq("Yes").astype(int)

    return out


# Nominal variables have no assumed numerical ordering.
NOMINAL_FEATURES = [
    "Job_Title",
    "Industry",
    "Company_Size",
    "Location",
    "Required_Skills",
]

# Numeric and explicitly ordered variables.
NUMERIC_FEATURES = [
    "Salary_USD",
    "ai_adoption_ord",
    "growth_ord",
    "remote_flag",
]


MODEL_FEATURES = NOMINAL_FEATURES + NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing to be fitted inside the ML Pipeline."""

    return ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                NOMINAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
