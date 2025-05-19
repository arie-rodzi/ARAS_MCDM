# ARAS Method Web App (Streamlit)

This repository provides a Streamlit-based web application for decision analysis using the **ARAS** (Additive Ratio Assessment) method.

---

## 🚀 Features

- Upload `.xlsx` file with:
  - Decision matrix
  - Weights
  - Criteria types (benefit/cost)
- Automatic normalization, weighting, scoring
- Step-by-step result display
- Downloadable Excel output

---

## 📁 Required Excel Format

### Sheet 1: `DecisionMatrix`

|     | C1 | C2 | C3 |
|-----|----|----|----|
| A1  | 10 | 5  | 8  |
| A2  | 8  | 7  | 6  |
| A3  | 9  | 6  | 7  |

### Sheet 2: `Weights`

| C1  | C2  | C3  |
|-----|-----|-----|
| 0.4 | 0.3 | 0.3 |

> Must sum to 1

### Sheet 3: `Types`

| C1     | C2   | C3     |
|--------|------|--------|
| benefit| cost | benefit|

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/aras-app.git
cd aras-app
pip install -r requirements.txt
streamlit run app_aras_with_tables.py
```

---

## 📦 Dependencies

- streamlit
- pandas
- numpy
- openpyxl
- xlsxwriter

---

## 📤 Output

- Normalized matrix
- Weighted matrix
- Final utility scores and rankings
- Excel report downloadable

---

## 👤 Author

- Zahari Md Rodzi (UiTM)
- 📧 zahari@uitm.edu.my