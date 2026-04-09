---
description: "Project guidelines for exponential regression modeling of student focus decline based on social media usage. Covers data preprocessing, model fitting (parametric form y=a·e^(bx)), evaluation metrics (R²≥0.40 minimum), code organization, and academic reporting standards."
---

# 📋 Exponential Regression Project Instructions

**Project**: Pemodelan Penurunan Tingkat Fokus Belajar Mahasiswa Berdasarkan Durasi Penggunaan Media Sosial Menggunakan Regresi Eksponensial

## 🎯 Core Objective

Build an exponential regression model **`focus_score = a · e^(b · social_media_hours)`** that predicts student learning focus decline as a function of daily social media usage. Variables:
- **Independent (X)**: `social_media_hours` — daily social media usage (hours)
- **Dependent (Y)**: `focus_score` — student focus level (0–100 scale estimated)
- **Target R²**: ≥ 0.40 minimum; ideal ≥ 0.65

---

## ⚠️ CRITICAL DATA HANDLING RULES

**Before ANY model fitting, you MUST validate target variables**:

1. **Filter `focus_score` strictly**: `focus_score > 0` (logarithm only works for positive values)
   ```python
   df = df[df['focus_score'] > 0]
   ```

2. **Filter `social_media_hours` validity**: `social_media_hours >= 0` (cannot be negative)
   ```python
   df = df[df['social_media_hours'] >= 0]
   ```

3. **Handle missing values BEFORE filtering**:
   ```python
   df = df.dropna(subset=['social_media_hours', 'focus_score'])
   ```

4. **Detect and remove outliers** using IQR method on both variables
   - Apply **before** model fitting, not after
   - Document outlier removal statistics (how many rows removed, thresholds used)

⚠️ **Forgetting this causes**: "math domain error", silent NaN predictions, invalid logarithms

---

## 🧮 MODELING METHODOLOGY

### Required: Two Fitting Approaches (Compare Both)

#### **Method 1: Linearization + Least Squares** (baseline)
Transform `y = a·e^(bx)` → `ln(y) = ln(a) + bx`, then regress:
```python
ln_Y = np.log(Y)  # Transform to linear space
coeffs = np.polyfit(X, ln_Y, 1)  # Returns [b, ln(a)]
b = coeffs[0]
a = np.exp(coeffs[1])
```

#### **Method 2: SciPy Curve Fitting** (recommended, more robust)
```python
from scipy.optimize import curve_fit

def exponential_func(x, a, b):
    return a * np.exp(b * x)

popt, pcov = curve_fit(exponential_func, X, Y, 
                       p0=[max(Y), -0.1],  # Initial guess
                       maxfev=10000)
a_opt, b_opt = popt
```

**Comparison table required**: Show both methods' R², RMSE, and parameter values side-by-side.

---

## 📊 EVALUATION METRICS (Non-Negotiable)

For every model, calculate and report:

| Metric | Formula | Acceptance Threshold |
|--------|---------|----------------------|
| **R²** | 1 − (SS_res / SS_tot) | ≥ 0.40 minimum, ≥ 0.65 ideal |
| **RMSE** | √(1/n Σ(y_actual − y_pred)²) | Document value (context-dependent) |
| **MAE** | mean(\|y_actual − y_pred\|) | Document value |
| **Residual properties** | Residuals ~ N(0, σ²) visually | Must show residual plot |

**If R² < 0.35**: Model is insufficient; consider:
  - Binning data by `social_media_hours` brackets (apply within-bin means)
  - Segment analysis (by gender/age group)
  - Adding control variables (multiple regression)
  - More aggressive outlier removal

---

## 📁 CODE ORGANIZATION & STRUCTURE

Required project layout:
```
project-regresi-eksponensial-anum/
├── data/
│   └── dataset.csv                    # ±20,000 rows; UTF-8 encoding
├── notebooks/
│   └── analisis_regresi.ipynb         # Jupyter notebook (reproducible)
├── src/
│   ├── preprocessing.py              # df cleaning, filtering, outlier removal
│   ├── modeling.py                   # both fitting methods (Method 1 & 2)
│   └── visualization.py              # plots: scatter+curve, residual, heatmap
├── output/
│   ├── figures/                      # PNG/PDF: saved at 150 DPI
│   │   ├── regresi_eksponensial.png
│   │   ├── residual_plot.png
│   │   └── heatmap_korelasi.png
│   └── hasil_model.txt               # Summary file: a, b, R², RMSE values
├── laporan/
│   └── laporan_akhir.docx            # Academic report (≥10 pages)
└── README.md
```

**Code standards**:
- All Python scripts must be **reproducible** (no random seeds left unseeded)
- Notebook cells **must execute sequentially** without errors
- Include **docstrings** for custom functions
- Document data decisions (why you chose specific thresholds for filtering)

---

## 📈 VISUALIZATION REQUIREMENTS

**Mandatory plots** (saved to `output/figures/` at 150 DPI):

1. **Scatter + Curve Overlay**
   - X-axis: `social_media_hours`
   - Y-axis: `focus_score`
   - Show actual points (alpha=0.3, small size)
   - Overlay fitted exponential curve (red line, width=2.5)
   - Legend with equation: `y = {a:.2f}·e^({b:.4f}x)`

2. **Residual Plot**
   - X-axis: predicted values (`Y_pred`)
   - Y-axis: residuals (`Y - Y_pred`)
   - Horizontal line at 0 (reference)
   - Assess randomness: should show no systematic pattern

3. **Correlation Heatmap**
   - All numeric columns included
   - Highlight correlation between `social_media_hours` and `focus_score`
   - Annotated with correlation coefficients (2 decimal places)

**Optional but recommended**:
   - Distribution histograms (social_media_hours, focus_score)
   - Box plots by hourly brackets
   - Bar chart of mean focus_score by media usage category (low/medium/high)

---

## 💬 PARAMETER INTERPRETATION (Must Document)

Write analytical narrative covering:

1. **Baseline focus score** (`a` value): "When `social_media_hours = 0`, the model predicts `focus_score = {a:.2f}`"
2. **Decay rate** (`b` value): "Focus score decreases by a factor of `e^{b:.4f}` ≈ `.{decay_pct:.1f}%` per additional hour of media usage"
3. **Critical point analysis**: "Focus score reaches critical level (e.g., 50) at `social_media_hours ≈ {x_critical:.2f}` hours"
   ```python
   x_critical = np.log(threshold / a) / b
   ```

---

## 📄 ACADEMIC REPORT STRUCTURE

**Minimum length**: 10 pages (excluding appendix)

```
BAB I — PENDAHULUAN
  1.1 Latar Belakang (problem context + statistics)
  1.2 Rumusan Masalah (research question)
  1.3 Tujuan (main + sub-objectives)
  1.4 Manfaat (practical applications)
  1.5 Batasan Masalah (scope, limitations)

BAB II — LANDASAN TEORI
  2.1 Fokus Belajar Mahasiswa (literature on academic focus)
  2.2 Media Sosial & Dampak Akademik (evidence of distraction)
  2.3 Regresi Eksponensial (mathematical foundation, forms)
  2.4 Metrik Evaluasi (R², RMSE, MAE explanations)

BAB III — METODOLOGI
  3.1 Dataset (source, n=20,000, columns overview)
  3.2 Tahapan Preprocessing (filtering, outlier removal, decisions)
  3.3 Teknik Pemodelan (Method 1 & 2 side-by-side)
  3.4 Tools & Environment (Python, library versions)

BAB IV — HASIL & PEMBAHASAN
  4.1 EDA Summary (distributions, correlations, patterns)
  4.2 Model Parameters (a, b values + interpretations)
  4.3 Akurasi Model (R², RMSE, MAE, comparison table)
  4.4 Visualisasi (graphs + narrative interpretation)
  4.5 Analisis Mendalam (critical points, implications, limitations)

BAB V — PENUTUP
  5.1 Kesimpulan (findings summary, model validity)
  5.2 Saran (future work, practical recommendations)

DAFTAR PUSTAKA (≥3 academic sources minimum)
LAMPIRAN (Python code snippets, extended tables)
```

**Equation formatting in report**:
```
f(x) = a · e^(bx)

Dengan:
- f(x) = focus_score (variabel dependen)
- x = social_media_hours (variabel independen, jam/hari)
- a = {nilai dari hasil fitting}
- b = {nilai dari hasil fitting, nilai negatif diharapkan}
- e = bilangan Euler (≈ 2.71828)
```

---

## ✅ DELIVERY CHECKLIST

- [ ] **Data**: Dataset loaded, validated (n ≥ 18,000 after cleaning)
- [ ] **Preprocessing**: Missing values, outliers handled; decisions documented
- [ ] **EDA**: Heatmap + scatter plot generated
- [ ] **Method 1**: Linearization fitting complete; R² calculated
- [ ] **Method 2**: SciPy curve_fit complete; R² calculated
- [ ] **Comparison**: Table showing both methods' results
- [ ] **Prediction**: Y_pred generated on original (or test subset)
- [ ] **Metrics**: MAE, MSE, RMSE, R² all calculated and reported
- [ ] **Visualizations**: 3 mandatory plots + 1+ optional plots; saved to `output/figures/`
- [ ] **Interpretation**: Parameter meanings documented narratively
- [ ] **Code**: Notebook/scripts reproducible; no hardcoded paths
- [ ] **Report**: ≥10 pages with all BAB sections; equations formatted professionally
- [ ] **Appendix**: Key code snippets included in report
- [ ] **Final deliverable**: All outputs in `output/` and `laporan/`

---

## 🚨 COMMON PITFALLS TO AVOID

| Pitfall | Fix |
|---------|-----|
| "ValueError: math domain error" during ln() | Filter `focus_score > 0` BEFORE linearization |
| Curve fitting "maxfev exceeded" or no convergence | Check initial guess `p0=[max(Y), -0.1]`; expand `maxfev` to 50000 |
| R² = 0.15 (far too low) | Data may not follow exponential pattern; try binning, segmentation, or multiple regression |
| Residual plot shows clear trend | Model is biased; consider polynomial or spline fit instead of exponential |
| "focus_score" has extreme outliers (0, 100+) | Validate data source; apply stricter IQR filtering (1.5→1.0) |
| Graphs doesn't appear in notebook | Ensure `%matplotlib inline` at top; use `plt.show()` |
| Report lacks academic rigor | Add citations from Google Scholar; reference dataset context |

---

## 🔗 REFERENCE LINKS IN PROJECT

- **Full specification**: Review `INSTRUKSI_PROJECT.pdf` or original instructions document for complete methodology
- **Dataset location**: `data/dataset.csv`
- **Python libraries**: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn (install via `pip install -r requirements.txt` if available)

## 📞 When in Doubt

1. **Data handling doubt?** → Reread the filtering rules (section with red warning box)
2. **Model fitting doubt?** → Implement BOTH Method 1 and Method 2, compare
3. **Visualization doubt?** → Follow the mandatory plots checklist exactly
4. **Report doubt?** → Use the BAB structure as rigid template; fill each section systematically

---

**Last Updated**: April 2026  
**Project Status**: In Development  
**Mata Kuliah**: Pemodelan & Simulasi  
**NIM**: 247007111152
