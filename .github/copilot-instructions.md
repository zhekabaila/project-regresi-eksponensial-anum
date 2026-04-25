---
description: 'Numerical Analysis project: exponential regression modeling of student focus decline using linearization and least squares method. Core focus on mathematical derivation of normal equations, parametric form y=C·e^(bx), RMS error calculation, and Indonesian language requirements for all outputs and code documentation.'
---

# 📋 INSTRUKSI PROJECT ANALISA NUMERIK

## Pemodelan Penurunan Tingkat Fokus Belajar Mahasiswa Berdasarkan Durasi Penggunaan Media Sosial Menggunakan Regresi Eksponensial

**Mata Kuliah**: **Analisa Numerik (Numerical Analysis)**  
**Topik**: Regresi Non-Linear — Regresi Eksponensial  
**Metode Utama**: Linearisasi Persamaan Eksponensial + Metode Kuadrat Terkecil  
**NIM**: 247007111152  
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
- **C** & **b**: Parameter yang dihitung melalui **linearisasi + persamaan normal**
- **Galat RMS (RMSE)**: Ukuran kualitas model utama (dari Analisa Numerik)
- **Target RMSE**: Serendah mungkin; R² ≥ 0.40 minimum sebagai indikator kelayakan

### Sub-Objectives:

1. Memahami dan menerapkan **linearisasi**: `y = C·e^(bx)` → `ln(y) = ln(C) + bx`
2. Menurunkan **persamaan normal** dari metode kuadrat terkecil secara analitik
3. Menghitung parameter `C` dan `b` dari data menggunakan **rumus analitis**
4. Mengukur kualitas model dengan **RMSE (Galat RMS), MAE, MSE, R²**
5. Visualisasi hasil dengan **Data Aktual (Y) vs Kurva Prediksi (Ŷ)** pada satu grafik
6. Seluruh output (teks cetak, judul grafik, label, komentar code) menggunakan **Bahasa Indonesia**

---

## ⚠️ CRITICAL DATA VALIDATION (WAJIB — Analisa Numerik)

**Sebelum melakukan proses linearisasi, WAJIB validasi data**:

### 1. Filter `focus_score > 0` (CRITICAL untuk Analisa Numerik)

```python
# focus_score HARUS positif karena akan ditransformasi dengan ln(y)
df = df[df['focus_score'] > 0]
```

**Alasan**: Fungsi logaritma natural hanya terdefinisi untuk nilai positif. Jika `focus_score <= 0`, akan terjadi `RuntimeWarning: invalid value in log` dan menghasilkan NaN.

### 2. Filter `social_media_hours >= 0`

```python
df = df[df['social_media_hours'] >= 0]  # Tidak boleh negatif
```

### 3. Hapus Missing Values TERLEBIH DAHULU

```python
df = df.dropna(subset=['social_media_hours', 'focus_score'])
```

### 4. Penanganan Outlier dengan IQR

Terapkan SEBELUM fitting:

```python
Q1 = df['focus_score'].quantile(0.25)
Q3 = df['focus_score'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['focus_score'] >= lower) & (df['focus_score'] <= upper)]
```

**Dokumentasikan**: Berapa banyak row dihapus di setiap tahap (laporan harus mencantumkan ini).

---

## 🧮 METODE PEMODELAN — ANALISA NUMERIK

### METODE 1: Linearisasi + Persamaan Normal (WAJIB — Inti Analisa Numerik)

Ini adalah metode yang **harus dikerjakan dan dipahami** karena merupakan inti materi Analisa Numerik.

#### Langkah 1: Transformasi Logaritma (Linearisasi)

```python
# Model asli: y = C * e^(b*x)
# Logaritmisasi: ln(y) = ln(C) + b*x
# Substitusi: Y' = ln(y), a = ln(C)
# Bentuk linear: Y' = a + b*x

Y_prime = np.log(Y)  # Y' = ln(Y)
```

#### Langkah 2: Hitung Komponen Persamaan Normal

```python
# Diperlukan untuk sistem 2 persamaan:
n       = len(X)
sum_x   = np.sum(X)
sum_Yp  = np.sum(Y_prime)
sum_x2  = np.sum(X**2)
sum_xYp = np.sum(X * Y_prime)
```

#### Langkah 3: Selesaikan Persamaan Normal (Rumus Analitis)

```python
# Sistem persamaan normal:
#   n*a + Σx*b = ΣY'
#   Σx*a + Σx²*b = ΣxY'
#
# Solusi (rumus analitis):
b = (n * sum_xYp - sum_x * sum_Yp) / (n * sum_x2 - sum_x**2)
a = (sum_Yp - b * sum_x) / n

# Kembalikan ke parameter asli:
C = np.exp(a)  # C = e^a
```

**WAJIB tampilkan**: Nilai n, Σx, ΣY', Σx², ΣxY' di laporan (tabel komponen persamaan normal)

### METODE 2: SciPy curve_fit (Pembanding)

```python
from scipy.optimize import curve_fit

def fungsi_eksponensial(x, C, b):
    return C * np.exp(b * x)

popt, pcov = curve_fit(fungsi_eksponensial, X, Y,
                       p0=[max(Y), -0.1], maxfev=10000)
C_scipy, b_scipy = popt
```

**WAJIB buat tabel perbandingan**: Kedua metode dengan metrik C, b, RMSE, MAE, R² berdampingan

---

## 📊 METRIK EVALUASI — ANALISA NUMERIK

### Metrik Wajib (untuk setiap model):

| Metrik               | Formula               | Interpretasi                                                  |
| -------------------- | --------------------- | ------------------------------------------------------------- |
| **RMSE (Galat RMS)** | √(1/n Σ(yᵢ - ŷᵢ)²)    | **Metrik utama Analisa Numerik** — semakin kecil semakin baik |
| **MAE**              | (1/n)·Σ\|yᵢ - ŷᵢ\|    | Rata-rata error absolut (dalam satuan Y)                      |
| **MSE**              | (1/n)·Σ(yᵢ - ŷᵢ)²     | MSE = RMSE²                                                   |
| **R²**               | 1 − (SS_res / SS_tot) | Proporsi variansi yang dijelaskan (0–1); ≥0.40 minimum        |

### Interpretasi RMSE dalam Konteks Analisa Numerik:

- **RMSE < 5**: Sangat baik (error kecil)
- **5 ≤ RMSE < 10**: Baik (error sedang)
- **RMSE ≥ 10**: Cukup (pertimbangkan model alternatif)

### Jika R² < 0.35:

- Data mungkin tidak cocok dengan model eksponensial
- Coba binning X dan rata-rata Y per bin
- Bandingkan dengan model linear atau polinomial
- Lakukan segmentasi data (analisis per subgroup)

---

## 💬 INTERPRETASI PARAMETER — KONTEKS ANALISA NUMERIK

### 1. Parameter C (Konstanta Awal)

```
C = e^a = [nilai dari hasil fitting]
```

**Interpretasi**:

- Nilai `focus_score` saat `social_media_hours = 0` (baseline, tanpa media sosial)
- Artinya: "Tanpa penggunaan media sosial, tingkat fokus mahasiswa diprediksi sebesar {C:.2f}"

### 2. Parameter b (Laju Perubahan Eksponensial)

```
b = [nilai dari hasil fitting, diharapkan negatif]
```

**Interpretasi**:

- Jika `b < 0`: Fokus MENURUN secara eksponensial seiring bertambahnya media sosial
- Faktor perubahan per jam: `e^b = [nilai]`
- Artinya: "Setiap penambahan 1 jam media sosial, fokus berkali dengan faktor {e^b:.4f} dari nilai sebelumnya"
- Atau: "Fokus berkurang sekitar {(1-e^b)\*100:.1f}% per jam tambahan media sosial"

### 3. Analisis Titik Kritis

```python
# Pada jam berapa fokus_score mencapai level kritis (misalnya 50)?
x_kritis = np.log(threshold / C) / b
```

**Interpretasi**:

- "Tingkat fokus mencapai level {threshold} pada saat penggunaan media sosial sekitar {x_kritis:.2f} jam per hari"
- "Rekomendasi: Mahasiswa harus membatasi penggunaan media sosial tidak lebih dari {x_kritis:.1f} jam per hari untuk menjaga fokus"

---

## 📄 STRUKTUR LAPORAN ANALISA NUMERIK

**Minimum**: 10 halaman (tidak termasuk cover & daftar isi)

### BAB I — PENDAHULUAN

- 1.1 Latar Belakang (fenomena media sosial, dampak fokus belajar)
- 1.2 Rumusan Masalah (bagaimana hubungan?, bagaimana membangun model Analisa Numerik?)
- 1.3 Tujuan
- 1.4 Manfaat
- 1.5 Batasan Masalah

### BAB II — LANDASAN TEORI (≥2 halaman)

**PENTING — Derivasi matematis WAJIB ada di sini:**

- 2.1 Konsep Regresi dalam Analisa Numerik (pencocokan kurva, linear vs non-linear)
- 2.2 Regresi Eksponensial (bentuk: y = C·e^(bx), aplikasi)
- **2.3 LINEARISASI PERSAMAAN EKSPONENSIAL** (derivasi lengkap)
  ```
  y = C·e^(bx)
  ln(y) = ln(C) + b·x
  Y' = a + b·x    [dimana Y' = ln(y), a = ln(C)]
  ```
- **2.4 METODE KUADRAT TERKECIL**
  - Fungsi error: R = Σ(Y'ᵢ - a - bxᵢ)²
  - Penurunan (turunan parsial ∂R/∂a, ∂R/∂b)
- **2.5 PERSAMAAN NORMAL**
  - Sistem 2 persamaan linear
  - Rumus analitis: b = ..., a = ...
- 2.6 Galat RMS (RMSE) sebagai metrik Analisa Numerik
- 2.7 Metrik tambahan (MAE, R²)

### BAB III — METODOLOGI

- 3.1 Dataset (sumber, n=20.000, deskripsi kolom)
- 3.2 Variabel (X = social_media_hours, Y = focus_score)
- 3.3 Tahapan Preprocessing (filtering, outlier removal — dengan jumlah row dihapus)
- **3.4 Algoritma Regresi Eksponensial** (flowchart atau pseudocode)
- 3.5 Tools (Python, library: pandas, numpy, scipy, sklearn, matplotlib)

### BAB IV — HASIL & PEMBAHASAN (±3 halaman, PALING PENTING)

- 4.1 Eksplorasi Data (statistik deskriptif, distribusi, korelasi)
- 4.2 Proses Linearisasi (tabel komponen persamaan normal: n, Σx, ΣY', Σx², ΣxY')
- **4.3 Hasil Persamaan Normal** (tampilkan sistem persamaan dengan nilai nyata, proses solusi)
- 4.4 Parameter Model (C, b dengan interpretasi)
- 4.5 Perbandingan Metode Linearisasi vs SciPy (tabel evaluasi)
- 4.6 Evaluasi Akurasi (RMSE, MAE, R² dengan penafsiran)
- **4.7 Visualisasi** (semua gambar dari output/figures/ dengan caption)
  - Scatter + kurva regresi
  - Proses linearisasi (sebelum/sesudah)
  - Residual plot
  - **Data Aktual (Y) vs Kurva Prediksi (Ŷ)** — WAJIB ada!
- 4.8 Analisis Titik Kritis
- 4.9 Pembahasan (makna parameter, validitas model)

### BAB V — PENUTUP

- 5.1 Kesimpulan (apakah model berhasil? Apakah R² ≥ 0.40?)
- 5.2 Saran (rekomendasi praktis, penelitian lanjutan)

### DAFTAR PUSTAKA

Minimal 4 sumber dari buku Analisa Numerik:

- Chapra, S.C. & Canale, R.P. (2015). _Numerical Methods for Engineers_
- Munir, R. (2010). _Metode Numerik_
- Burden, R.L. & Faires, J.D. (2011). _Numerical Analysis_
- Walpole et al. (2012). _Probability & Statistics for Engineers_

### LAMPIRAN

- Kode Python lengkap
- Tabel komponen persamaan normal (subset 10-20 titik data)
- Grafik tambahan

---

## ✅ CHECKLIST PROGRES PROJECT

### Fase Validasi Data

- [ ] `focus_score > 0` difilter (CRITICAL untuk logaritma)
- [ ] `social_media_hours >= 0` difilter
- [ ] Missing values dihapus
- [ ] Outlier dihapus dengan IQR
- [ ] n ≥ 18.000 setelah cleaning

### Fase Analisa Numerik — Metode 1 (Linearisasi)

- [ ] `Y' = ln(Y)` dihitung
- [ ] Komponen persamaan normal dihitung (n, Σx, ΣY', Σx², ΣxY')
- [ ] Parameter `b` dihitung dengan rumus analitis
- [ ] Parameter `a` dihitung
- [ ] Parameter `C = e^a` dihitung
- [ ] Persamaan diformulasikan: `focus_score = C · e^(b · social_media_hours)`
- [ ] RMSE dihitung untuk Metode 1
- [ ] MAE, R² dihitung

### Fase Metode 2 (Pembanding)

- [ ] SciPy `curve_fit` dijalankan
- [ ] RMSE, MAE, R² dihitung untuk Metode 2
- [ ] Tabel perbandingan (C, b, RMSE, MAE, R²) dibuat

### Fase Visualisasi (Wajib Bahasa Indonesia)

- [ ] Scatter + kurva regresi (dengan label "Data Aktual" dan "Kurva Prediksi")
- [ ] **Grafik Data Aktual (Y) vs Kurva Prediksi (Ŷ)** berdampingan (WAJIB)
- [ ] Grafik proses linearisasi sebelum/sesudah
- [ ] Residual plot
- [ ] Heatmap korelasi
- [ ] Semua judul & label grafik berbahasa Indonesia
- [ ] Semua disimpan di `output/figures/` (DPI 150)

### Fase Dokumentasi Kode

- [ ] Semua komentar Python berbahasa Indonesia
- [ ] Nama variabel deskriptif berbahasa Indonesia
- [ ] Output `print()` berbahasa Indonesia
- [ ] File `output/hasil_model.txt` tersimpan

### Fase Laporan (Analisa Numerik)

- [ ] BAB I–II selesai dengan derivasi matematis lengkap
- [ ] BAB III–IV selesai dengan tabel komponen persamaan normal
- [ ] Tabel perbandingan Metode 1 vs 2 dimasukkan
- [ ] Semua grafik dimasukkan ke laporan dengan caption
- [ ] Persamaan y = C·e^(bx) ditulis formal di laporan
- [ ] Daftar pustaka (≥4 sumber Analisa Numerik)
- [ ] Laporan ≥10 halaman
- [ ] Laporan disimpan di `laporan/laporan_akhir.docx`

---

## ⚠️ POTENSI MASALAH & SOLUSI

| Masalah                                         | Penyebab                                           | Solusi                                                          |
| ----------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------- |
| `RuntimeError: invalid value in log`            | Ada `focus_score <= 0` yang tidak difilter         | Filter `df[df['focus_score'] > 0]` TERLEBIH DAHULU sebelum ln() |
| `curve_fit` tidak konvergen                     | Initial guess `p0` buruk atau maxfev terlalu kecil | Coba `p0=[max(Y), -0.5]` atau `maxfev=50000`                    |
| R² < 0.35 (jauh di bawah threshold)             | Data mungkin tidak mengikuti pola eksponensial     | Bandingkan dengan model linear; coba binning X                  |
| RMSE sangat besar (> 20)                        | Model underfitting atau ada outlier ekstrem        | Terapkan IQR filtering lebih ketat (gunakan 1.0 bukan 1.5)      |
| Parameter `b` positif (tidak sesuai ekspektasi) | Data tidak menunjukkan tren penurunan              | Periksa scatter plot awal; validasi data source                 |
| Residual plot menunjukkan pola sistematis       | Model bias atau pola non-eksponensial dominan      | Pertimbangkan polynomial fit sebagai pembanding                 |
| Laporan tidak mencantumkan derivasi matematis   | Lupa bahwa ini adalah mata kuliah ANALISA NUMERIK  | Tambahkan derivasi persamaan normal lengkap di BAB II           |
| Grafik menggunakan Bahasa Inggris               | Instruksi project mensyaratkan Bahasa Indonesia    | Ubah semua label ke Bahasa Indonesia                            |

---

## 🇮🇩 KETENTUAN BAHASA (WAJIB)

> **Semua output program dan dokumentasi kode HARUS menggunakan Bahasa Indonesia.**

### 1. Output Program (teks yang ditampilkan ke console)

```python
# ✅ BENAR
print("Model berhasil dilatih.")
plt.title('Tingkat Fokus Belajar vs Durasi Media Sosial')
plt.xlabel('Durasi Penggunaan Media Sosial (jam/hari)')

# ❌ SALAH
print("Model trained successfully.")
plt.title('Focus Score vs Social Media Hours')
plt.xlabel('Social Media Hours (hours/day)')
```

### 2. Komentar dalam Kode Python

```python
# ✅ BENAR
# Hitung jumlah data
jumlah_data = len(X)

# Transformasi logaritma natural untuk linearisasi
nilai_ln_y = np.log(Y)

# ❌ SALAH
# Calculate number of data points
n_data = len(X)

# Log transformation
log_y = np.log(Y)
```

### 3. Nama Variabel Deskriptif

```python
# ✅ BENAR (untuk variabel deskriptif di domain project)
komponen_persamaan_normal = {...}
parameter_C = ...
parameter_b = ...
galat_RMS = ...

# ❌ SALAH
normal_eq_components = {...}
param_C = ...
param_b = ...
error_RMS = ...

# CATATAN: Nama fungsi/metode library tetap English (np.log, pd.read_csv, plt.scatter)
```

---

## 🔗 REFERENSI BOOK ANALISA NUMERIK (Wajib Dibaca untuk Memahami Project)

1. **Chapra, S.C. & Canale, R.P.** (2015). _Numerical Methods for Engineers_. 7th Edition. McGraw-Hill.
   - Chapter 17–18: Least Squares Regression
2. **Munir, R.** (2010). _Metode Numerik_. Informatika Bandung.
   - Chapter 7: Pencocokan Kurva (Curve Fitting)
3. **Burden, R.L. & Faires, J.D.** (2011). _Numerical Analysis_. 9th Edition. Brooks/Cole.
   - Chapter 8: Approximation Theory

4. **Walpole, R.E., Myers, R.H., Myers, S.L.** (2012). _Probability & Statistics for Engineers and Scientists_. 9th Edition. Pearson.
   - Chapter 12: Regression

---

## 📅 ESTIMASI WAKTU PENGERJAAN

| Fase                                           | Estimasi       |
| ---------------------------------------------- | -------------- |
| Setup environment + EDA + preprocessing        | 3–4 jam        |
| Metode 1 (linearisasi manual + rumus analitis) | 2–3 jam        |
| Metode 2 (SciPy) + evaluasi perbandingan       | 1 jam          |
| Visualisasi lengkap (5 grafik)                 | 2 jam          |
| Interpretasi & analisis                        | 1–2 jam        |
| Penulisan laporan dengan derivasi matematis    | 6–8 jam        |
| **Total Estimasi**                             | **~15–20 jam** |

---

**Versi**: 2.0 — Analisa Numerik Focus  
**Last Updated**: April 2026  
**Mata Kuliah**: Analisa Numerik (Numerical Analysis)  
**Topik Khusus**: Regresi Eksponensial — Linearisasi + Kuadrat Terkecil  
**NIM**: 247007111152
