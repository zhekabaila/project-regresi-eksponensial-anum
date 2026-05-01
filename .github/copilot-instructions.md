---
description: 'Numerical Analysis project: exponential regression modeling of student focus decline using linearization and least squares method. Core focus on mathematical derivation of normal equations, parametric form y=C·e^(bx), RMS error calculation, and Indonesian language requirements for all outputs and code documentation.'
---

# 📋 INSTRUKSI PROJECT ANALISA NUMERIK

## Pemodelan Penurunan Tingkat Fokus Belajar Mahasiswa Berdasarkan Durasi Penggunaan Media Sosial Menggunakan Regresi Eksponensial

**Mata Kuliah**: **Analisa Numerik (Numerical Analysis)**  
**Topik**: Regresi Non-Linear — Regresi Eksponensial  
**Metode Utama**: Linearisasi Persamaan Eksponensial + Metode Kuadrat Terkecil  
**NIM**: 247007111152  
**Sumber Data**: Kaggle  
**Project**: Pemodelan Penurunan Tingkat Fokus Belajar Mahasiswa Berdasarkan Durasi Penggunaan Media Sosial Menggunakan Regresi Eksponensial

---

## 🎯 OBJECTIVE UTAMA — Konteks Analisa Numerik

Menerapkan konsep **Analisa Numerik** — khususnya teknik **linearisasi persamaan non-linear** dan **metode kuadrat terkecil (least squares)** — untuk membangun model regresi eksponensial:

```
focus_score = C · e^(b · social_media_hours)
```

Dengan:

- **X** (independen): `social_media_hours` — durasi penggunaan media sosial per hari (jam)
- **Y** (dependen): `focus_score` — tingkat fokus belajar mahasiswa
- **C** & **b**: Parameter yang dihitung melalui **linearisasi + persamaan normal (least squares)**
- **Galat RMS (RMSE)**: Ukuran kualitas model utama (dari Analisa Numerik)
- **Target R²**: Minimum 0.40 sebagai indikator kelayakan model

### Sub-Objectives:

1. Memahami dan menerapkan **linearisasi**: `y = C·e^(bx)` → `ln(y) = ln(C) + bx`
2. Menurunkan **persamaan normal** dari metode kuadrat terkecil secara analitik
3. Menghitung parameter `C` dan `b` dari data menggunakan **rumus analitis (Metode 1)**
4. Membandingkan dengan metode numerik robust **SciPy Curve Fitting (Metode 2)**
5. Mengukur kualitas model dengan **RMSE, MAE, MSE, R²**
6. Visualisasi hasil dengan **Data Aktual vs Kurva Prediksi** pada grafik scatter
7. Seluruh output (teks cetak, judul grafik, label, komentar code) menggunakan **Bahasa Indonesia**

---

## 📊 DATASET

- **Sumber**: Kaggle
- **Variabel Independen (X)**: `social_media_hours` (jam/hari)
- **Variabel Dependen (Y)**: `focus_score` (skor fokus belajar)
- **Variabel Kontrol**: `age`, `gender`, `sleep_hours`, `stress_level`, `study_hours_per_day`, dan lainnya
- **Format**: CSV dengan header
- **Lokasi**: `data/dataset.csv`

---

## 📁 STRUKTUR FOLDER

```
project-regresi-eksponensial-anum/
├── data/
│   └── dataset.csv                    # Dataset dari Kaggle
├── notebooks/
│   └── analisis_regresi.ipynb         # Jupyter Notebook analisis lengkap
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Preprocessing & data validation
│   ├── modeling.py                    # Regresi eksponensial (Metode 1 & 2)
│   └── visualization.py               # Plotting & visualisasi
├── output/
│   ├── figures/                       # Grafik hasil (PNG, DPI 150)
│   └── hasil_model.txt                # Ringkasan hasil model
├── laporan/
│   └── laporan_akhir.docx             # Laporan akademik (≥10 halaman)
├── copilot-instructions.md            # Instruksi ini
└── README.md
```

---

## 🔄 PIPELINE ANALISIS — LOGIC IMPLEMENTASI SAAT INI

### TAHAP 1: Data Loading & Validation

**Fungsi**: `load_and_validate_dataset(filepath: str) -> pd.DataFrame`

**Operasi**:

- Load CSV dataset
- Validasi kolom yang diperlukan: `'social_media_hours'` dan `'focus_score'`
- Raise `ValueError` jika ada kolom yang hilang
- Print status loading

**Output**: DataFrame terload

---

### TAHAP 2: Exploratory Data Analysis (EDA)

**Tujuan**: Memahami distribusi dan karakteristik data sebelum preprocessing

**Komponen Analisis**:

- Statistik deskriptif (min, max, mean, std)
- Distribusi X dan Y (histogram)
- Heatmap korelasi antar variabel
- Scatter plot raw data
- Perhitungan korelasi Pearson

**Visualisasi**:

- `plot_perbandingan_distribusi(X_raw, Y_raw)` — Side-by-side histogram
- `plot_heatmap_korelasi(df_raw, output_path)` — Correlation heatmap
- Manual scatter plot dengan nilai korelasi

---

### TAHAP 3: Data Preprocessing — Cleaning Pipeline Sequential

**Fungsi Master**: `preprocess_pipeline(filepath: str, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]`

**Tahapan Sequential**:

#### 3.1 Remove Missing Values

**Fungsi**: `remove_missing_values(df, subset=['social_media_hours', 'focus_score'])`

- Hapus baris dengan NaN di kolom kunci
- Print: jumlah baris dihapus dan persentase

#### 3.2 Filter Valid Values (CRITICAL untuk Analisa Numerik)

**Fungsi**: `filter_valid_values(df)`

**Rules**:

- `focus_score > 0` **(CRITICAL)** — Diperlukan untuk `ln(y)`
- `social_media_hours >= 0` — Tidak boleh negatif

**Alasan**: Fungsi logaritma natural hanya terdefinisi untuk nilai positif. Jika `focus_score <= 0`, akan terjadi `RuntimeWarning: invalid value in log` → NaN values.

#### 3.3 Remove Outliers dengan IQR Method

**Fungsi**: `detect_outliers_iqr(df, columns=['social_media_hours', 'focus_score'], iqr_multiplier=1.5)`

**Algoritma**:

```
Untuk setiap kolom:
  Q1 = quantile(0.25)
  Q3 = quantile(0.75)
  IQR = Q3 - Q1
  Lower = Q1 - 1.5*IQR
  Upper = Q3 + 1.5*IQR
  Hapus jika value < Lower atau value > Upper
```

**Output**:

- Print batas untuk setiap kolom
- Print jumlah outlier dihapus
- Dataframe dengan outlier dihapus

**Pipeline Output**:

- X: numpy array social_media_hours (cleaned)
- Y: numpy array focus_score (cleaned)
- df_clean: cleaned dataframe
- Print: statistik lengkap sebelum/sesudah

---

### TAHAP 4 & 5: Exponential Regression Fitting — Dual Method Approach

**Fungsi Master**: `bandingkan_metode(X: np.ndarray, Y: np.ndarray) -> Dict`

Menjalankan kedua metode fitting, evaluasi, dan membandingkan hasilnya.

#### METODE 1: Linearisasi + Persamaan Normal (Inti Analisa Numerik)

**Fungsi**: `metode1_linearisasi(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, np.ndarray]`

**Langkah Matematis**:

1. **Transformasi Logaritma (Linearisasi)**:

   ```
   Model asli:       y = C * e^(b*x)
   Logaritmisasi:    ln(y) = ln(C) + b*x
   Substitusi:       Y' = ln(y), a = ln(C)
   Bentuk linear:    Y' = a + b*x
   ```

2. **Hitung Komponen Persamaan Normal**:

   ```
   n = len(X)
   Σx = sum(X)
   ΣY' = sum(ln(Y))
   Σx² = sum(X²)
   ΣxY' = sum(X * ln(Y))
   ```

3. **Gunakan np.polyfit untuk Regresi Linear**:

   ```python
   Y_prime = np.log(Y)
   koefisien = np.polyfit(X, Y_prime, 1)  # Returns [b, ln(C)]
   b = koefisien[0]
   C = np.exp(koefisien[1])
   ```

4. **Prediksi**:
   ```python
   Y_pred = fungsi_eksponensial(X, C, b)  # Y_pred = C * exp(b*X)
   ```

**Output Informasi**:

- Persamaan model: `y = C * e^(b*x)` dengan nilai C dan b
- Komponen persamaan normal: n, Σx, ΣY', Σx², ΣxY'
- Nilai prediksi Y_pred

**Return**: `(C, b, Y_pred)`

---

#### METODE 2: SciPy Curve Fitting dengan Algoritma Levenberg-Marquardt

**Fungsi**: `metode2_scipy_curve_fit(X: np.ndarray, Y: np.ndarray, p0: list = None, maxfev: int = 10000) -> Tuple[float, float, np.ndarray]`

**Algoritma Optimisasi**: Levenberg-Marquardt (robust non-linear optimization)

```python
def fungsi_eksponensial(x, C, b):
    return C * np.exp(b * x)

popt, pcov = curve_fit(
    fungsi_eksponensial, X, Y,
    p0=[max(Y), -0.1],  # Initial guess
    maxfev=10000        # Max function evaluations
)

C, b = popt
Y_pred = fungsi_eksponensial(X, C, b)
return C, b, Y_pred
```

**Keuntungan Metode 2**:

- Lebih robust terhadap initial guess
- Handling non-linear optimization secara superior
- Fallback: jika maxfev gagal, retry dengan `maxfev*5`

**Return**: `(C, b, Y_pred)`

---

### TAHAP 6: Model Evaluation & Comparison

**Fungsi**: `evaluasi_model(Y_aktual: np.ndarray, Y_prediksi: np.ndarray, nama_model: str = "") -> Dict[str, float]`

**Metrik Evaluasi**:

| Metrik   | Formula               | Interpretasi                       |
| -------- | --------------------- | ---------------------------------- |
| **MAE**  | (1/n)·Σ\|y - ŷ\|      | Error rata-rata absolut            |
| **MSE**  | (1/n)·Σ(y - ŷ)²       | Mean squared error                 |
| **RMSE** | √MSE                  | **UTAMA untuk Analisa Numerik**    |
| **R²**   | 1 - (SS_res / SS_tot) | Proporsi variansi dijelaskan (0-1) |

**Assessment Otomatis**:

- R² ≥ 0.65: ✓ Sangat baik
- 0.40 ≤ R² < 0.65: ✓ Dapat diterima
- R² < 0.40: ⚠ Pertimbangkan penyempurnaan model

**Return**: `{'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}`

---

**Fungsi**: `bandingkan_metode(X, Y) -> Dict`

**Operasi**:

1. Jalankan `metode1_linearisasi(X, Y)` → evaluasi → metrik1
2. Jalankan `metode2_scipy_curve_fit(X, Y)` → evaluasi → metrik2
3. Print tabel perbandingan: C, b, MAE, RMSE, R² untuk kedua metode
4. Tentukan metode terbaik berdasarkan R² tertinggi
5. Return hasil komprehensif

**Return Structure**:

```python
{
    'metode1': {
        'C': C1,
        'b': b1,
        'Y_pred': Y_pred1,
        'metrik': {'MAE': ..., 'MSE': ..., 'RMSE': ..., 'R2': ...}
    },
    'metode2': {
        'C': C2,
        'b': b2,
        'Y_pred': Y_pred2,
        'metrik': {'MAE': ..., 'MSE': ..., 'RMSE': ..., 'R2': ...}
    }
}
```

---

### TAHAP 7: Visualization — 5 Output Wajib

#### Plot 1: Regression Curve with Data

**Fungsi**: `plot_regresi_dengan_data(X, Y, Y_pred, C, b, output_path, title)`

- Scatter plot: data aktual
- Overlay: prediction curve `y = C·e^(b·x)`
- Legend: "Data Aktual" vs "Kurva Prediksi: y = C·e^(b·x)"
- Output: PNG, DPI 150

#### Plot 2: Residual Analysis

**Fungsi**: `plot_residual(Y, Y_pred, output_path, title)`

- Scatter: (Y_pred, Y - Y_pred)
- Garis horizontal reference: y = 0
- Statistik residual: Mean, Std, Min, Max

#### Plot 3: Correlation Heatmap

**Fungsi**: `plot_heatmap_korelasi(df, output_path, title)`

- Correlation matrix heatmap
- Annotasi nilai korelasi
- Highlight korelasi dengan focus_score

#### Plot 4: Distribution Histograms

**Fungsi**: `plot_perbandingan_distribusi(X, Y)`

- Side-by-side histograms
- X: social_media_hours distribution
- Y: focus_score distribution
- Bins: 50, dengan grid

#### Plot 5: Focus Score by Usage Bracket

**Fungsi**: `plot_fokus_per_bracket(df, output_path)`

- Group data: [0-2j], [2-4j], [4-6j], [6-8j], [8-10j], [10j+]
- Bar chart: rata-rata focus_score per bracket
- Error bars: std deviation
- Tabel statistik: mean, std, count per bracket

**Persyaratan Visualisasi**:

- ✅ Semua judul & label dalam **Bahasa Indonesia**
- ✅ Tersimpan di `output/figures/` dengan DPI 150
- ✅ Readable di laporan (font size ≥ 10pt)

---

### TAHAP 8: Critical Point Analysis & Parameter Interpretation

**Fungsi**: `hitung_titik_kritis(C: float, b: float, threshold: float = 50) -> float`

**Rumus Matematis**:

```
Cari x saat y mencapai threshold:
threshold = C * e^(b*x)
ln(threshold) = ln(C) + b*x
x = [ln(threshold) - ln(C)] / b
x = ln(threshold / C) / b
```

**Interpretasi Parameter**:

| Parameter       | Makna                                                 |
| --------------- | ----------------------------------------------------- |
| **C**           | Focus score saat social_media_hours = 0 (baseline)    |
| **b**           | Koefisien eksponensial (biasanya negatif = penurunan) |
| **e^b**         | Faktor pengali fokus per jam (biasanya < 1)           |
| **Penurunan %** | (1 - e^b) × 100 = penurunan persentase fokus per jam  |

**Output Deskriptif**:

- "Parameter C = ... : Fokus baseline adalah ... (saat 0 jam media sosial)"
- "Parameter b = ... : Setiap jam tambahan, fokus berkali e^(...) ≈ ... (penurunan ...%)"
- "Titik kritis: Fokus mencapai level 50 pada ... jam penggunaan"

---

### TAHAP 9: Save Model Results

**Fungsi**: `simpan_hasil_model(filepath: str, C: float, b: float, metrik: Dict, jumlah_data: int)`

**Format Output** (`output/hasil_model.txt`):

```
HASIL MODEL REGRESI EKSPONENSIAL
═════════════════════════════════════════════════

PERSAMAAN MODEL
─────────────────────────────────────────────────
focus_score = [C] · e^([b] · social_media_hours)

PARAMETER
─────────────────────────────────────────────────
C = [nilai]
b = [nilai]
Jumlah data = [n]

METRIK EVALUASI
─────────────────────────────────────────────────
MAE = [nilai]
MSE = [nilai]
RMSE = [nilai]
R² = [nilai]

═════════════════════════════════════════════════
Dihasilkan: April 2026
```

---

## ⚠️ CRITICAL DATA VALIDATION RULES (WAJIB)

**Sebelum linearisasi, WAJIB validasi**:

### 1. Filter `focus_score > 0` (CRITICAL)

Alasan: `ln(y)` hanya terdefinisi untuk y > 0

### 2. Filter `social_media_hours >= 0`

Alasan: Durasi tidak boleh negatif

### 3. Remove Missing Values Terlebih Dahulu

Sebelum filtering dan outlier removal

### 4. IQR Outlier Detection

```python
Q1 = quantile(0.25)
Q3 = quantile(0.75)
IQR = Q3 - Q1
Lower = Q1 - 1.5 * IQR
Upper = Q3 + 1.5 * IQR
# Hapus jika value < Lower atau > Upper
```

**Dokumentasikan**: Setiap tahap preprocessing, print jumlah baris dihapus

---

## 💬 OUTPUT LANGUAGE — Bahasa Indonesia (WAJIB)

### Code Comments

```python
# ✅ BENAR
# Transformasi logaritma untuk linearisasi
Y_prime = np.log(Y)

# ❌ SALAH
# Log transformation
log_y = np.log(Y)
```

### Print Output

```python
# ✅ BENAR
print("METODE 1: LINEARISASI + PERSAMAAN NORMAL")
print(f"Persamaan: y = {C:.4f} * e^({b:.6f} * x)")

# ❌ SALAH
print("METHOD 1: LINEARIZATION + NORMAL EQUATIONS")
```

### Visualization Labels

```python
# ✅ BENAR
plt.xlabel('Durasi Penggunaan Media Sosial (jam/hari)')
plt.ylabel('Skor Fokus')
plt.title('Regresi Eksponensial')

# ❌ SALAH
plt.xlabel('Social Media Hours')
plt.title('Exponential Regression')
```

---

## 📄 LAPORAN AKADEMIK — Struktur Minimum (≥10 halaman)

### BAB I: Pendahuluan (±2 hal)

- Latar belakang masalah
- Rumusan masalah
- Tujuan penelitian
- Manfaat penelitian

### BAB II: Landasan Teori (±2 hal)

- Regresi eksponensial: y = C·e^(bx)
- **Linearisasi persamaan eksponensial** (derivasi lengkap)
- **Metode Kuadrat Terkecil**: Fungsi error, penurunan ∂/∂a = 0 dan ∂/∂b = 0
- **Persamaan Normal** (sistem 2 persamaan linear)
- Metrik evaluasi: RMSE, R²

### BAB III: Metodologi (±2 hal)

- Dataset dari Kaggle (deskripsi, variabel)
- **Tahapan Preprocessing**: Remove missing, filter, IQR outlier removal
- **Algoritma Regresi**: Metode 1 (Linearisasi) dan Metode 2 (SciPy)
- Tools: Python, pandas, numpy, scipy, sklearn, matplotlib

### BAB IV: Hasil & Pembahasan (±3 hal)

- **4.1 Eksplorasi Data**: Statistik, distribusi, korelasi, grafik
- **4.2 Hasil Preprocessing**: Tabel rows setiap tahap, range nilai
- **4.3 Fitting Results**: Persamaan, tabel perbandingan, grafik
- **4.4 Evaluasi Akurasi**: Metrik MAE/RMSE/R², assessment
- **4.5 Interpretasi Parameter**: Makna C, b, persen penurunan
- **4.6 Analisis Titik Kritis**: Kapan fokus mencapai level 50?

### BAB V: Penutup (±1 hal)

- Kesimpulan
- Saran dan penelitian lanjutan

---

## ✅ CHECKLIST IMPLEMENTASI

### Data Processing

- [x] Load dataset dari Kaggle
- [x] Validasi kolom yang diperlukan
- [x] Preprocessing pipeline (missing, filter, outlier)
- [x] Dokumentasi jumlah rows per tahap

### Modeling

- [x] Metode 1: Linearisasi + np.polyfit
- [x] Hitung komponen persamaan normal
- [x] Metode 2: SciPy curve_fit
- [x] Evaluasi kedua metode (MAE, RMSE, R²)
- [x] Bandingkan dan pilih terbaik

### Visualization

- [x] Scatter + prediction curve
- [x] Residual plot
- [x] Heatmap korelasi
- [x] Distribution histograms
- [x] Focus by bracket bar chart
- [x] Semua ke `output/figures/` DPI 150
- [x] Semua label Bahasa Indonesia

### Reporting

- [ ] BAB I–V lengkap
- [ ] Derivasi persamaan normal di BAB II
- [ ] Semua grafik dengan caption
- [ ] ≥4 referensi akademik
- [ ] Lampiran: kode Python

---

## ⚡ QUICK START

```bash
cd /Users/zhekabaila/project-regresi-eksponensial-anum
source venv/bin/activate
jupyter notebook notebooks/analisis_regresi.ipynb
# Jalankan: Cell → Run All
```

---

**Versi**: 2.2 — Updated sesuai logic kode saat ini  
**Sumber Data**: Kaggle  
**Last Updated**: Mei 2026  
**Status**: Production Ready
