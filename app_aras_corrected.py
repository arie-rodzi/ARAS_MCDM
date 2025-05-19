import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="ARAS Method App (Corrected)", layout="wide")
st.title("ARAS Method (Additive Ratio Assessment) - Verified Implementation")

st.markdown("Upload an Excel file with three sheets:")
st.markdown("1. `DecisionMatrix` — Alternatives × Criteria values")
st.markdown("2. `Weights` — One row of weights (must sum to 1)")
st.markdown("3. `Types` — One row of 'benefit' or 'cost' labels per criterion")

uploaded_file = st.file_uploader("📤 Upload Excel file", type=["xlsx"])

if uploaded_file:
    try:
        # Read and prepare data
        dm = pd.read_excel(uploaded_file, sheet_name="DecisionMatrix", index_col=0)
        weights_df = pd.read_excel(uploaded_file, sheet_name="Weights", header=0)
        types_df = pd.read_excel(uploaded_file, sheet_name="Types", header=0)

        weights = pd.to_numeric(weights_df.values[0], errors='coerce')
        types = [str(x).strip().lower() for x in types_df.values[0]]

        # Validate input
        if np.isnan(weights).any():
            st.error("❌ One or more weights are non-numeric or missing.")
            st.stop()
        if any(t not in {"benefit", "cost"} for t in types):
            st.error("❌ Criteria types must be 'benefit' or 'cost'.")
            st.stop()
        if len(weights) != dm.shape[1] or len(types) != dm.shape[1]:
            st.error("❌ Weights and types must match number of criteria.")
            st.stop()
        if not np.isclose(np.sum(weights), 1.0):
            st.error("❌ Weights must sum to 1.")
            st.stop()

        st.subheader("Step 1: Raw Decision Matrix")
        st.dataframe(dm)

        # Step 2: Add Ideal Alternative
        ideal = {}
        for j, t in enumerate(types):
            if t == "benefit":
                ideal[dm.columns[j]] = dm.iloc[:, j].max()
            else:
                ideal[dm.columns[j]] = dm.iloc[:, j].min()

        dm_ideal = pd.concat([pd.DataFrame([ideal], index=["Ideal"]), dm])
        st.subheader("Step 2: Matrix with Ideal Alternative")
        st.dataframe(dm_ideal)

        # Step 3: Normalize
        norm_matrix = dm_ideal.copy()
        for j, t in enumerate(types):
            col = norm_matrix.columns[j]
            if t == "benefit":
                norm_matrix[col] = norm_matrix[col] / norm_matrix.loc["Ideal", col]
            else:
                norm_matrix[col] = norm_matrix.loc["Ideal", col] / norm_matrix[col]

        st.subheader("Step 3: Normalized Matrix (ARAS)")
        st.dataframe(norm_matrix)

        # Step 4: Weighted Normalized Matrix
        weighted_matrix = norm_matrix * weights
        st.subheader("Step 4: Weighted Normalized Matrix")
        st.dataframe(weighted_matrix)

        # Step 5: Utility Scores
        S = weighted_matrix.sum(axis=1)
        S0 = S["Ideal"]
        Ku = S / S0

        result_df = pd.DataFrame({
            "Utility Score (S)": S,
            "Relative Utility (Kᵢ = Sᵢ / S₀)": Ku,
            "Rank": Ku.rank(ascending=False).astype(int)
        }, index=dm_ideal.index)

        st.subheader("Step 5: Final Scores and Ranking")
        st.markdown(f"**S₀ (Ideal Alternative Score)**: `{S0:.4f}`")
        st.dataframe(result_df.sort_values("Rank"))

        # Export all to Excel
        def to_excel():
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                dm.to_excel(writer, sheet_name="Input_Data")
                pd.DataFrame([weights], columns=dm.columns).to_excel(writer, sheet_name="Weights", index=False)
                pd.DataFrame([types], columns=dm.columns).to_excel(writer, sheet_name="Types", index=False)
                dm_ideal.to_excel(writer, sheet_name="Matrix_with_Ideal")
                norm_matrix.to_excel(writer, sheet_name="Normalized")
                weighted_matrix.to_excel(writer, sheet_name="Weighted")
                result_df.to_excel(writer, sheet_name="ARAS_Result")
            output.seek(0)
            return output

        excel_data = to_excel()
        st.download_button("📥 Download Full Report (Excel)", data=excel_data,
                           file_name="aras_corrected_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"❌ Error: {e}")