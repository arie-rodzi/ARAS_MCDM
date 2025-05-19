import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="ARAS Method App", layout="wide")
st.title("ARAS Method (Additive Ratio Assessment)")

st.markdown("Upload an Excel file with three sheets:")
st.markdown("1. `DecisionMatrix` — alternatives × criteria values")
st.markdown("2. `Weights` — a single row with weights (must sum to 1)")
st.markdown("3. `Types` — a single row of 'benefit' or 'cost' labels for each criterion")

uploaded_file = st.file_uploader("📤 Upload Excel file", type=["xlsx"])

if uploaded_file:
    try:
        # Load data
        dm = pd.read_excel(uploaded_file, sheet_name="DecisionMatrix", index_col=0)
        weights_df = pd.read_excel(uploaded_file, sheet_name="Weights", header=0)
        types_df = pd.read_excel(uploaded_file, sheet_name="Types", header=0)

        weights = pd.to_numeric(weights_df.values[0], errors='coerce')
        types = [str(x).strip().lower() for x in types_df.values[0]]

        # Validation
        if np.isnan(weights).any():
            st.error("❌ One or more weights are non-numeric or missing.")
            st.stop()

        valid_types = {"benefit", "cost"}
        if any(t not in valid_types for t in types):
            st.error("❌ One or more criteria types are not 'benefit' or 'cost'.")
            st.stop()

        if len(weights) != dm.shape[1] or len(types) != dm.shape[1]:
            st.error("❌ Number of weights/types must match the number of criteria.")
            st.stop()

        if not np.isclose(np.sum(weights), 1.0):
            st.error("❌ Weights must sum to 1. Current sum: {:.4f}".format(np.sum(weights)))
            st.stop()

        # Step 1: Show Raw Decision Matrix
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
        st.subheader("Step 2: Decision Matrix with Ideal Alternative")
        st.dataframe(dm_ideal)

        # Step 3: Normalize
        norm_matrix = dm_ideal.copy()
        for j, t in enumerate(types):
            col = norm_matrix.columns[j]
            if t == "benefit":
                norm_matrix[col] = norm_matrix[col] / norm_matrix.loc["Ideal", col]
            else:
                norm_matrix[col] = norm_matrix.loc["Ideal", col] / norm_matrix[col]

        st.subheader("Step 3: Normalized Decision Matrix")
        st.dataframe(norm_matrix)

        # Step 4: Weighted Normalized Matrix
        weighted_norm = norm_matrix * weights
        st.subheader("Step 4: Weighted Normalized Matrix")
        st.dataframe(weighted_norm)

        # Step 5: Utility Scores
        utility_scores = weighted_norm.sum(axis=1)
        optimal_score = utility_scores.loc["Ideal"]
        relative_utility = utility_scores / optimal_score

        result_df = pd.DataFrame({
            "Utility Score": utility_scores,
            "Relative Utility": relative_utility,
            "Rank": relative_utility.rank(ascending=False).astype(int)
        }, index=dm_ideal.index)

        st.subheader("Step 5: Final Scores and Ranks")
        st.dataframe(result_df.sort_values("Rank"))

        # Export to Excel
        def to_excel():
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                dm.to_excel(writer, sheet_name="Input_Data")
                pd.DataFrame([weights], columns=dm.columns).to_excel(writer, sheet_name="Weights", index=False)
                pd.DataFrame([types], columns=dm.columns).to_excel(writer, sheet_name="Types", index=False)
                dm_ideal.to_excel(writer, sheet_name="Matrix_with_Ideal")
                norm_matrix.to_excel(writer, sheet_name="Normalized")
                weighted_norm.to_excel(writer, sheet_name="Weighted")
                result_df.to_excel(writer, sheet_name="ARAS_Result")
            output.seek(0)
            return output

        excel_data = to_excel()
        st.download_button("📥 Download All Results (Excel)", data=excel_data,
                           file_name="aras_step_by_step.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"❌ Error: {e}")