import streamlit as st
import pandas as pd
import pdfplumber
import re
import plotly.express as px
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Drilling Fluid Report Dashboard", layout="wide")
st.title("📈 Drilling Fluid Report Dashboard")

uploaded_files = st.file_uploader("Upload Daily Drilling Fluid Reports (PDFs)", type="pdf", accept_multiple_files=True)

@st.cache_data
def extract_data_from_pdf(pdf_file):
    data = []
    with pdfplumber.open(pdf_file) as pdf:
        full_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

        # Extract date
        date_match = re.search(r"Date\s+(\d{4}-\d{2}-\d{2})", full_text)
        date = date_match.group(1) if date_match else None

        # Table-based parsing for more accurate values
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                try:
                    flat_table = "\n".join([" ".join([str(cell) if cell is not None else "" for cell in row]) for row in table if row])
                    full_text += "\n" + flat_table
                except Exception:
                    continue

        # Extract mud weight
        mw_match = re.search(r"Density.*?(\d{2}\.\d).*?ppg", full_text)
        mud_weight = float(mw_match.group(1)) if mw_match else None

        # Extract PV, YP
        pv_match = re.search(r"PV.*?(\d{1,2})\s*@", full_text)
        yp_match = re.search(r"YP.*?(\d{1,2})\s", full_text)
        pv = int(pv_match.group(1)) if pv_match else None
        yp = int(yp_match.group(1)) if yp_match else None

        # Extract ES (Electrical Stability)
        es_match = re.search(r"Elec\. Stability V\s+(\d+)", full_text)
        es = int(es_match.group(1)) if es_match else None

        # Extract losses and additions
        add_match = re.findall(r"Additions bbl\s+([\d\.]+).*?Total Losses\s+([\d\.]+)", full_text, re.DOTALL)
        if add_match:
            additions, losses = map(float, add_match[0])
        else:
            additions = losses = 0.0

        if date:  # Only add row if valid date exists
            data.append({
                "Date": date,
                "Mud Weight": mud_weight,
                "PV": pv,
                "YP": yp,
                "Electrical Stability": es,
                "Additions (bbl)": additions,
                "Losses (bbl)": losses
            })
    return pd.DataFrame(data)

if uploaded_files:
    combined_df = pd.DataFrame()
    status_msgs = []
    progress_bar = st.progress(0)
    with st.spinner("Processing uploaded PDF reports..."):
        for i, file in enumerate(uploaded_files):
            try:
                file_data = extract_data_from_pdf(file)
                if not file_data.empty:
                    combined_df = pd.concat([combined_df, file_data], ignore_index=True)
                    status_msgs.append(f"✅ {file.name} processed successfully.")
                else:
                    status_msgs.append(f"⚠️ {file.name} skipped (no valid data extracted).")
            except Exception as e:
                status_msgs.append(f"❌ {file.name} failed: {str(e)}")
            progress_bar.progress((i + 1) / len(uploaded_files))

    for msg in status_msgs:
        st.write(msg)

    if not combined_df.empty:
        st.success(f"✅ Finished processing {len(uploaded_files)} report(s).")

        combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
        combined_df = combined_df.sort_values("Date")

        st.subheader("Extracted Summary Table")
        st.dataframe(combined_df)

        st.subheader("🌍 Mud Properties Over Time")
        fig1 = px.line(combined_df, x="Date", y=["Mud Weight", "PV", "YP"], markers=True)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("🚧 Additions vs. Losses")
        fig2 = px.bar(combined_df, x="Date", y=["Additions (bbl)", "Losses (bbl)"], barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🔌 Electrical Stability")
        fig3 = px.line(combined_df, x="Date", y="Electrical Stability", markers=True)
        st.plotly_chart(fig3, use_container_width=True)

        # Machine Learning Section
        st.subheader("🤖 ML Insights")
        df_ml = combined_df.dropna()
        if len(df_ml) > 5:
            X = df_ml[["Mud Weight", "PV", "YP", "Electrical Stability"]]

            # --- Loss Forecasting ---
            st.markdown("### 🔮 Loss Prediction")
            y_loss = df_ml["Losses (bbl)"]
            X_train, X_test, y_train, y_test = train_test_split(X, y_loss, test_size=0.2, random_state=42)
            loss_model = RandomForestRegressor(n_estimators=100, random_state=42)
            loss_model.fit(X_train, y_train)
            y_pred_loss = loss_model.predict(X_test)
            st.markdown(f"**Loss RMSE**: {np.sqrt(mean_squared_error(y_test, y_pred_loss)):.2f} bbl")
            st.markdown(f"**Loss R²**: {r2_score(y_test, y_pred_loss):.2f}")
            st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": y_pred_loss}))

            # --- YP Optimization ---
            st.markdown("### 🧪 YP Optimization Model")
            y_yp = df_ml["YP"]
            X_train_yp, X_test_yp, y_train_yp, y_test_yp = train_test_split(X.drop(columns=["YP"]), y_yp, test_size=0.2, random_state=42)
            yp_model = LinearRegression()
            yp_model.fit(X_train_yp, y_train_yp)
            y_pred_yp = yp_model.predict(X_test_yp)
            st.markdown(f"**YP RMSE**: {np.sqrt(mean_squared_error(y_test_yp, y_pred_yp)):.2f}")
            st.markdown(f"**YP R²**: {r2_score(y_test_yp, y_pred_yp):.2f}")
            st.dataframe(pd.DataFrame({"Actual YP": y_test_yp, "Predicted YP": y_pred_yp}))
        else:
            st.info("Not enough complete records for ML modeling. Upload more reports.")
    else:
        st.warning("No valid data extracted from uploaded PDFs.")
else:
    st.info("Upload one or more drilling fluid report PDFs to begin.")
