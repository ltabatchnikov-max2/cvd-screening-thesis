# MCC-based uncertainty zone classification
# Patients with MCC < threshold are assigned to one of three groups:

# Clinical Review Group:
#   All three class probabilities are similar (no clear preference)
#   Patient is referred for individual physician consultation

# No-Action Group:
#   Uncertainty is between LOW and INTERMEDIARY (and reverse) only
#   Original prediction is kept

# Upclassification Group:
#   Uncertainty is between HIGH and INTERMEDIARY (and reverse) only
#   All predictions are reclassified to HIGH to avoid missing a high-risk case

from src.config import MCC_THRESHOLD, TARGET_CLASS, ZONE0_PROBA_GAP


def get_zone(row, proba_columns):
    """
    Assign a zone label to a single uncertain prediction.
    """
    p          = {col.replace("proba_", ""): row[col] for col in proba_columns}
    sorted_cls = sorted(p, key=p.get, reverse=True)
    top1, top2, top3 = sorted_cls

    if p[top2] - p[top3] < ZONE0_PROBA_GAP:
        return "zone0"

    pair = tuple(sorted([top1, top2]))
    if pair == ("INTERMEDIARY", "LOW"):
        return "zone1"

    return "zone2"


def apply_mcc_strategy(df, mcc_threshold=MCC_THRESHOLD, target_class=TARGET_CLASS):
    """
    Apply the MCC-based screening strategy to a predictions DataFrame.

    Patients with MCC < mcc_threshold are assigned to a zone.
    Zone 2 patients are upclassified to target_class (HIGH).
    Passing mcc_threshold=0.0 returns the DataFrame unchanged (standard screening).
    """
    df2 = df.copy()

    if mcc_threshold == 0.0:
        return df2

    proba_columns = [c for c in df2.columns if c.startswith("proba_")]
    below_mask    = df2["mcc"] < mcc_threshold

    df2.loc[below_mask, "zone"] = df2[below_mask].apply(
        lambda row: get_zone(row, proba_columns), axis=1
    )

    upward_mask = below_mask & (df2["zone"] == "zone2")
    df2.loc[upward_mask, "pred"] = target_class
    for col in proba_columns:
        cls = col.replace("proba_", "")
        df2.loc[upward_mask, col] = 1.0 if cls == target_class else 0.0

    return df2