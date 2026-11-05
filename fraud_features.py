
import numpy as np
import pandas as pd

def clean_key(series):
    return (
        series.astype("string")
        .fillna("Missing")
        .astype(str)
    )

def prepare_features(
    raw,
    stats,
    numeric_original,
    categorical,
    final_features
):
    df = raw.copy()

    for col in categorical:
        df[col] = clean_key(df[col])

    for col in numeric_original:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # İlk aşamadaki türetilmiş zaman ve tutar alanları
    dt = df["TransactionDT"].fillna(0)
    amt = df["TransactionAmt"]

    df["TransactionHour"] = (
        np.floor(dt / 3600) % 24
    ).astype("float32")

    df["TransactionDay"] = (
        np.floor(dt / 86400)
    ).astype("float32")

    df["TransactionWeek"] = (
        np.floor(dt / (86400 * 7))
    ).astype("float32")

    df["TransactionAmtLog"] = np.log1p(
        amt.clip(lower=0)
    ).astype("float32")

    df["TransactionCents"] = (
        np.rint(amt.fillna(0) * 100) % 100
    ).astype("float32")

    df["LargeTransaction"] = (
        amt > 200
    ).astype("int8")

    # Eksik bilgi sayıları
    df["missing_numeric_count"] = (
        df[numeric_original]
        .isna()
        .sum(axis=1)
        .astype("float32")
    )

    df["missing_category_count"] = (
        df[categorical]
        .eq("Missing")
        .sum(axis=1)
        .astype("float32")
    )

    # Kategori sıklıkları
    for col, counts in stats["frequencies"].items():
        keys = clean_key(df[col])

        df["freq_" + col] = (
            keys.map(counts)
            .fillna(0)
            .div(stats["n_rows"])
            .astype("float32")
        )

    # Kart-adres kombinasyonu
    combo = clean_key(df["card1"]).str.cat(
        clean_key(df["addr1"]),
        sep="|"
    )

    df["freq_card1_addr1"] = (
        combo.map(stats["combo_frequency"])
        .fillna(0)
        .div(stats["n_rows"])
        .astype("float32")
    )

    # Grup bazlı işlem tutarı istatistikleri
    amount = pd.to_numeric(
        df["TransactionAmt"], errors="coerce"
    )

    for col in stats["amount_means"]:
        keys = clean_key(df[col])

        mean_amount = keys.map(
            stats["amount_means"][col]
        ).fillna(stats["global_amount_mean"])

        std_amount = keys.map(
            stats["amount_stds"][col]
        ).fillna(stats["global_amount_std"])

        df["amount_ratio_" + col] = (
            amount / mean_amount.clip(lower=1)
        ).clip(0, 50).astype("float32")

        df["amount_zscore_" + col] = (
            (amount - mean_amount)
            / std_amount.clip(lower=1)
        ).clip(-50, 50).astype("float32")

    return df[final_features].replace(
        [np.inf, -np.inf], np.nan
    )
