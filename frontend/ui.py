import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Churn App", page_icon="💳", layout="wide")

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")

api_base = st.sidebar.text_input(
    "FastAPI base URL",
    value="http://localhost:8000",
    help="Change this if your backend is deployed elsewhere",
)

page = st.sidebar.radio(
    "Navigate",
    ["Single Prediction", "Bulk Predict (CSV)", "Model Dashboard"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Bank Churn Prediction • XGBoost + Random Forest")

# Helper to build full URL
def api_url(path: str) -> str:
    return api_base.rstrip("/") + path


# ---------- 1️⃣ Single Prediction ----------
if page == "Single Prediction":
    st.title("💳 Single Customer Churn Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        CreditScore = st.number_input("Credit Score", 300, 900, 600)
        Age = st.number_input("Age", 18, 100, 35)
        Tenure = st.number_input("Tenure (Years)", 0, 10, 5)
        NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])

    with col2:
        Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
        IsActiveMember = st.selectbox("Is Active Member?", [0, 1])

    with col3:
        Balance = st.number_input("Account Balance", 0.0, 250000.0, 50000.0)
        EstimatedSalary = st.number_input(
            "Estimated Salary", 0.0, 300000.0, 75000.0
        )

    input_data = {
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
    }

    if st.button("🔮 Predict"):
        with st.spinner("Contacting API..."):
            try:
                res = requests.post(api_url("/predict"), json=input_data)
                if res.status_code != 200:
                    st.error(f"API error: {res.status_code} - {res.text}")
                else:
                    result = res.json()
                    xgb = result["xgboost_churn_probability"]
                    rf = result["random_forest_churn_probability"]
                    avg = result.get("average_probability", (xgb + rf) / 2)

                    st.subheader("📊 Prediction Result")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("XGBoost Prob.", f"{xgb:.3f}")
                    c2.metric("Random Forest Prob.", f"{rf:.3f}")
                    c3.metric("Average Prob.", f"{avg:.3f}")

                    st.markdown("### 🧾 Verdict")
                    if avg > 0.5:
                        st.error("⚠️ Customer is **likely to churn**.")
                    else:
                        st.success("✅ Customer is **not likely to churn**.")
            except Exception as e:
                st.error(f"Request failed: {e}")


# ---------- 2️⃣ Bulk Predict ----------
elif page == "Bulk Predict (CSV)":
    st.title("📁 Bulk Churn Prediction from CSV")

    st.markdown(
        """
    **Instructions**
    - Upload a CSV with columns:  
      `CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary`
    - The app will call the API in batch and append three columns:
      - `xgb_prob`, `rf_prob`, `avg_prob`
    """
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Preview of uploaded data:")
        st.dataframe(df.head())

        required_cols = [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns in CSV: {missing}")
        else:
            
            if st.button("🚀 Run Bulk Prediction"):
                records = df[required_cols].to_dict(orient="records")
                payload = {"customers": records}

                with st.spinner("Calling batch prediction API..."):
                    try:
                        res = requests.post(api_url("/predict-batch"), json=payload)

                        if res.status_code != 200:
                            st.error(f"API error: {res.status_code} - {res.text}")
                        else:
                            out = res.json()

                            if "predictions" not in out:
                                st.error("Unexpected API response: missing 'predictions' key")
                            else:
                                pred_list = out["predictions"]
                                df_pred = pd.DataFrame(pred_list)

                                # Merge predictions with original CSV
                                df_final = pd.concat([df, df_pred], axis=1)

                                st.success("✅ Predictions generated!")
                                st.dataframe(df_final.head())

                                # CSV download
                                csv_out = df_final.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                "⬇️ Download CSV with Predictions",
                                data=csv_out,
                                    file_name="churn_predictions_with_probs.csv",
                                    mime="text/csv",
                                )

                    except Exception as e:
                        st.error(f"Request failed: {e}")



# ---------- 3️⃣ Dashboard ----------
elif page == "Model Dashboard":
    st.title("📊 Model Performance Dashboard")

    with st.spinner("Fetching metrics from API..."):
        try:
            res = requests.get(api_url("/metrics"))
        except Exception as e:
            res = None
            st.error(f"Request failed: {e}")

    if res is None:
        st.stop()

    if res.status_code != 200:
        st.error(f"API error: {res.status_code} - {res.text}")
    else:
        data = res.json()
        if "detail" in data:
            st.warning(data["detail"])
        else:
            roc = data["roc"]
            cms = data["confusion_matrix"]

            # --- ROC Curve ---
            st.subheader("ROC Curve")

            fpr_xgb = np.array(roc["xgboost"]["fpr"])
            tpr_xgb = np.array(roc["xgboost"]["tpr"])
            auc_xgb = roc["xgboost"]["auc"]

            fpr_rf = np.array(roc["random_forest"]["fpr"])
            tpr_rf = np.array(roc["random_forest"]["tpr"])
            auc_rf = roc["random_forest"]["auc"]

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.3f})")
            ax_roc.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

            # --- Confusion Matrices ---
            st.subheader("Confusion Matrices")
            cm_xgb = np.array(cms["xgboost"])
            cm_rf = np.array(cms["random_forest"])

            c1, c2 = st.columns(2)

            def plot_cm(cm, title):
                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation="nearest")
                ax.set_title(title)
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            j,
                            i,
                            str(cm[i, j]),
                            ha="center",
                            va="center",
                        )
                return fig

            with c1:
                st.markdown("#### XGBoost")
                st.pyplot(plot_cm(cm_xgb, "XGBoost Confusion Matrix"))

            with c2:
                st.markdown("#### Random Forest")
                st.pyplot(plot_cm(cm_rf, "Random Forest Confusion Matrix"))
