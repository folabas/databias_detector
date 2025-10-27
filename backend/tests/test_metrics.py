import pandas as pd
from backend.core.analysis import (
    compute_statistical_parity_difference,
    compute_disparate_impact_ratio,
    compute_predictive_equality_difference,
)

def test_metrics_run_on_simple_data():
    df = pd.DataFrame({
        "gender": ["M","F","M","F"],
        "accepted": [1,0,1,1],
        "pred": [1,0,1,0],
    })
    spd = compute_statistical_parity_difference(df, "gender", "accepted")
    diratio = compute_disparate_impact_ratio(df, "gender", "accepted")
    ped = compute_predictive_equality_difference(df, "gender", "accepted", "pred")
    assert isinstance(spd, float)
    assert isinstance(diratio, float)
    assert isinstance(ped, float)