
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from fraud_features import prepare_features

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="IEEE-CIS Fraud Detection",
    page_icon="🔎",
    layout="wide"
)

@st.cache_resource
def load_bundle():
    return joblib.load(
        ROOT / "ieee_fraud_final_model.joblib"
    )

bundle = load_bundle()

model = bundle["model"]
features = bundle["feature_columns"]
original = bundle["original_features"]
categorical = bundle["categorical_columns"]

numeric_original = [
    c for c in original
    if c not in categorical
]

stats = bundle["reference_stats"]

def predict_rows(df):
    X = prepare_features(
        df,
        stats,
        numeric_original,
        categorical,
        features
    )

    return model.predict_proba(X)[:, 1]

st.title("🔎 IEEE-CIS Fraud Detection")

st.write(
    "HistGradientBoosting + "
    "Fraud Feature Engineering"
)

c1, c2, c3 = st.columns(3)

c1.metric("Validation ROC-AUC", "0.912001")
c2.metric("Validation PR-AUC", "0.534559")
c3.metric("Model Features", len(features))

st.warning(
    "Bu uygulama eğitim amaçlıdır. "
    "Gerçek kart, kimlik, hesap veya kişisel "
    "işlem verileri yüklemeyin. "
    "Model skoru tek başına dolandırıcılık "
    "tespiti ya da yaptırım kararı değildir."
)

demo = pd.read_csv(
    ROOT / "synthetic_transaction.csv"
)

tab1, tab2 = st.tabs([
    "Yapay işlem simülasyonu",
    "CSV ile toplu tahmin"
])

with tab1:
    st.subheader("Örnek işlem özellikleri")

    st.caption(
        "Örnek profil, eğitim verisinin sütun "
        "istatistiklerinden oluşturulmuştur. "
        "Gerçek bir işlem değildir."
    )

    amount = st.number_input(
        "İşlem tutarı",
        min_value=0.0,
        max_value=1000000.0,
        value=float(demo["TransactionAmt"].iloc[0]),
        step=10.0
    )

    product_options = sorted(
        stats["frequencies"]["ProductCD"]
        .index.astype(str).tolist()
    )

    product = st.selectbox(
        "Ürün tipi (ProductCD)",
        product_options,
        index=(
            product_options.index(
                str(demo["ProductCD"].iloc[0])
            )
            if str(demo["ProductCD"].iloc[0])
            in product_options
            else 0
        )
    )

    card = st.text_input(
        "Anonim kart kategorisi (card1)",
        value=str(demo["card1"].iloc[0])
    )

    if st.button(
        "İşlem Skorunu Hesapla",
        type="primary"
    ):
        sample = demo.copy()

        sample.loc[0, "TransactionAmt"] = amount
        sample.loc[0, "ProductCD"] = product
        sample.loc[0, "card1"] = card

        score = float(predict_rows(sample)[0])

        st.metric(
            "Modelin ürettiği skor",
            f"{score:.4f}"
        )

        st.caption(
            "Skor 0-1 aralığındadır fakat "
            "kalibre edilmiş kesin dolandırıcılık "
            "olasılığı değildir."
        )

    st.download_button(
        "Yapay örnek CSV indir",
        data=demo.to_csv(index=False),
        file_name="synthetic_transaction.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("CSV ile toplu analiz")

    st.write(
        f"CSV dosyası {len(original)} özgün "
        "özellik sütununu içermelidir. "
        "16 ek özellik otomatik hesaplanır."
    )

    with st.expander("Gerekli sütunları göster"):
        st.code(", ".join(original))

    uploaded = st.file_uploader(
        "Önceden hazırlanmış özellik CSV'sini seç",
        type=["csv"]
    )

    if uploaded is not None:

        df = pd.read_csv(uploaded, nrows=2001)

        st.write("Okunan satır sayısı:", len(df))

        missing = [
            col for col in original
            if col not in df.columns
        ]

        if missing:
            st.error(
                "Eksik sütunlar: "
                + ", ".join(missing)
            )

        elif not 1 <= len(df) <= 2000:
            st.error(
                "Tek seferde en fazla 2.000 işlem kullan."
            )

        elif st.button(
            "Toplu Tahminleri Hesapla",
            type="primary"
        ):
            try:
                scores = predict_rows(df)

                result = pd.DataFrame({
                    "row_number": np.arange(
                        1, len(scores) + 1
                    ),
                    "model_score": scores
                })

                st.success("Tahminler tamamlandı.")
                st.dataframe(result.head(30))

                st.download_button(
                    "Sonuçları CSV indir",
                    data=result.to_csv(index=False),
                    file_name="ieee_fraud_predictions.csv",
                    mime="text/csv"
                )

            except Exception as exc:
                st.error(
                    "Veri veya model uyumluluğu hatası."
                )
                st.code(str(exc))

st.caption(
    "Educational ML project - "
    "Kaggle IEEE-CIS Fraud Detection"
)
