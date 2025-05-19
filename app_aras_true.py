import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Correct ARAS Method", layout="wide")
st.title("ARAS Method (Correct Ratio-Based Formulation)")

st.markdown("Upload an Excel file with:")
st.markdown("1. `DecisionMatrix`: alternatives × criteria values")
st.markdown("2. `Weights`: one row (must sum to 1)")
st.markdown("3. `Types`: one row of 'benefit' or 'cost'")

uploaded_file = st.file_uploader("📤 Upload Excel file", type=["xlsx"])

if uploaded_file:
    try:
        # Read sheets
        dm = pd.read_excel(uploaded_file, sheet_name="DecisionMatrix", index_col=0)
        weights_df = pd.read_excel(uploaded_file, sheet_name="Weights", header=0)
        types_df = pd.read_excel(uploaded_file, sheet_name="Types", header=0)

        weights = pd.to_numeric(weights_df.values[0], errors='coerce')
        types = [str(x).strip().lower() for x in types_df.values[0]]

        # Validation
        if np.isnan(weights).any():
            st.error("❌ Non-numeric weights found.")
            st.stop()
        if any(t not in {"benefit", "cost"} for t in types):
            st.error("❌ Types must be 'benefit' or 'cost'.")
            st.stop()
        if len(weights) != dm.shape[1] or len(types) != dm.shape[1]:
            st.error("❌ Criteria count mismatch in weights/types.")
            st.stop()
        if not np.isclose(sum(weights), 1.0):
            st.error("❌ Weights must sum to 1.")
            st.stop()

        st.subheader("Step 1: Raw Decision Matrix")
        st.dataframe(dm)

        # Step 2: Normalization
        norm_matrix = pd.DataFrame(index=dm.index, columns=dm.columns)
        for j, col in enumerate(dm.columns):
            if types[j] == "benefit":
                norm_matrix[col] = dm[col] / dm[col].sum()
            else:  # cost
                inv = 1 / dm[col]
                norm_matrix[col] = inv / inv.sum()

        st.subheader("Step 2: Normalized Matrix")
        st.dataframe(norm_matrix)

        # Step 3: Weighted Normalized Matrix
        weighted_matrix = norm_matrix * weights
        st.subheader("Step 3: Weighted Normalized Matrix")
        st.dataframe(weighted_matrix)

        # Step 4: Utility Score Qᵢ = S⁺ / S⁻
        benefit_idx = [i for i, t in enumerate(types) if t == "benefit"]
        cost_idx = [i for i, t in enumerate(types) if t == "cost"]

        S_pos = weighted_matrix.iloc[:, benefit_idx].sum(axis=1)
        S_neg = weighted_matrix.iloc[:, cost_idx].sum(axis=1)

        Q = S_pos / S_neg

        result_df = pd.DataFrame({
            "S⁺ (Benefit Sum)": S_pos,
            "S⁻ (Cost Sum)": S_neg,
            "Qᵢ = S⁺ / S⁻": Q,
            "Rank": Q.rank(ascending=False).astype(int)
        }, index=dm.index)

        st.subheader("Step 4: Final Utility Scores and Ranking")
        st.dataframe(result_df.sort_values("Rank"))

        # Export to Excel
        def to_excel():
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                dm.to_excel(writer, sheet_name="DecisionMatrix")
                pd.DataFrame([weights], columns=dm.columns).to_excel(writer, sheet_name="Weights", index=False)
                pd.DataFrame([types], columns=dm.columns).to_excel(writer, sheet_name="Types", index=False)
                norm_matrix.to_excel(writer, sheet_name="Normalized")
                weighted_matrix.to_excel(writer, sheet_name="Weighted")
                result_df.to_excel(writer, sheet_name="ARAS_Result")
            output.seek(0)
            return output

        excel_export = to_excel()
        st.download_button("📥 Download All Results (Excel)", data=excel_export,
                           file_name="aras_correct_ratio_based.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")