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

        # Custom Additions and Losses extraction
        base_match = re.search(r"Base\s+(\d+\.\d+)", full_text)
        water_match = re.search(r"Drill Water\s+(\d+\.\d+)", full_text)
        barite_match = re.search(r"Barite\s+(\d+\.\d+)", full_text)
        chem_match = re.search(r"Chemicals\s+(\d+\.\d+)", full_text)

        additions = sum([
            float(base_match.group(1)) if base_match else 0,
            float(water_match.group(1)) if water_match else 0,
            float(barite_match.group(1)) if barite_match else 0,
            float(chem_match.group(1)) if chem_match else 0
        ])

        sce_match = re.search(r"SCE\s+(\d+\.\d+)", full_text)
        misc_match = re.search(r"Misc Other\s+(\d+\.\d+)", full_text)

        losses = sum([
            float(sce_match.group(1)) if sce_match else 0,
            float(misc_match.group(1)) if misc_match else 0
        ])

        if date:
            data.append({
                "Date": date,
                "Mud Weight": mud_weight,
                "PV": pv,
                "YP": yp,
                "Electrical Stability": es,
                "Additions (bbl)": additions,
                "Losses (bbl)": losses,
                "Total Dilution (bbl)": additions - losses if additions > losses else 0,
                "Total SCE (bbl)": losses,
                "Mud Cutting Ratio": (losses / additions) if additions > 0 else None,
                "DSRE %": (losses / (additions + losses) * 100) if (additions + losses) > 0 else None
            })
    return pd.DataFrame(data)
