# Pemodelan Penurunan Tingkat Fokus Belajar Mahasiswa Berdasarkan Durasi Penggunaan Media Sosial Menggunakan Regresi Eksponensial

**Mata Kuliah**: Pemodelan & Simulasi  
**NIM**: 247007111152  
**Tanggal**: April 2026

## 🎯 Objektif

Membangun model regresi eksponensial yang memprediksi penurunan tingkat fokus belajar mahasiswa (`focus_score`) berdasarkan durasi penggunaan media sosial (`social_media_hours`) per hari.

**Model**: `focus_score = a · e^(b · social_media_hours)`

## 📊 Dataset

- **File**: `data/dataset.csv`
- **Baris**: ±20.000
- **Variabel Independen (X)**: `social_media_hours` (jam/hari)
- **Variabel Dependen (Y)**: `focus_score` (skor fokus)
- **Variabel Kontrol**: `age`, `gender`, `sleep_hours`, `stress_level`, dll.

## 📁 Struktur Folder

```
project-regresi-eksponensial-anum/
├── data/
│   └── dataset.csv                    # File dataset utama
├── notebooks/
│   └── analisis_regresi.ipynb         # Jupyter Notebook analisis
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Fungsi pembersihan & persiapan data
│   ├── modeling.py                    # Fungsi fitting (Method 1 & 2)
│   └── visualization.py               # Fungsi plotting & visualisasi
├── output/
│   ├── figures/                       # Grafik hasil (PNG/PDF 150 DPI)
│   └── hasil_model.txt                # Ringkasan hasil model
├── laporan/
│   └── laporan_akhir.docx             # Laporan akademik (≥10 halaman)
├── copilot-instructions.md            # AI assistant guidelines
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Setup Environment

```bash
# Buat virtual environment (optional tapi direkomendasikan)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Persiapkan Dataset

Letakkan file `dataset.csv` di folder `data/`:

```bash
cp /path/to/dataset.csv data/dataset.csv
```

### 3. Jalankan Analisis

Buka dan jalankan Jupyter Notebook:

```bash
jupyter notebook notebooks/analisis_regresi.ipynb
```

Atau jalankan sebagai Python script (jika sudah dikonversi):

```bash
python notebooks/analisis_regresi.py
```

## 📋 Tahapan Pengerjaan

- [x] Setup struktur folder
- [ ] Load dan cleaning dataset
- [ ] Exploratory Data Analysis (EDA)
- [ ] Metode 1: Linearisasi + Least Squares
- [ ] Metode 2: SciPy Curve Fitting
- [ ] Perbandingan metode
- [ ] Evaluasi model (MAE, MSE, RMSE, R²)
- [ ] Visualisasi hasil
- [ ] Penulisan laporan akademik

## 📌 Persyaratan Minimum (Acceptance Criteria)

- **R² ≥ 0.40** (ideal: ≥ 0.65)
- **3 grafik wajib**: scatter+kurva, residual plot, heatmap korelasi
- **Laporan ≥ 10 halaman** dengan 5 BAB standar
- **Kode Python reproducible** tanpa hardcoded path
- **Kedua metode fitting** diimplementasikan dan dibandingkan

## 🔗 Dokumentasi & Guidelines

Baca **`copilot-instructions.md`** untuk:

- Aturan data handling kritis
- Metodologi pemodelan detail
- Struktur laporan akademik
- Checklist deliverable
- Troubleshooting common pitfalls

## 📚 Library yang Digunakan

- `pandas` — data manipulation
- `numpy` — numerical computation
- `scipy` — curve fitting, optimization
- `scikit-learn` — metrics (MAE, MSE, R²)
- `matplotlib` — visualization
- `seaborn` — statistical plots
- `jupyter` — interactive notebooks

## 📞 Catatan Penting

- **Selalu filter** `focus_score > 0` sebelum logaritmik
- **Gunakan kedua metode fitting**, bandingkan hasilnya
- **Dokumentasikan setiap keputusan** (outlier threshold, binning, filtering)
- **Simpan semua grafik** di `output/figures/` dengan DPI 150

## 📄 Laporan Akhir

Laporan harus mencakup:

1. **BAB I**: Pendahuluan (background, problem statement, objectives)
2. **BAB II**: Landasan Teori (fokus, media sosial, regresi eksponensial)
3. **BAB III**: Metodologi (dataset, preprocessing, teknik fitting)
4. **BAB IV**: Hasil & Pembahasan (EDA, model parameters, akurasi, grafik)
5. **BAB V**: Penutup (kesimpulan, saran)

File laporan: `laporan/laporan_akhir.docx` atau `.pdf`

---

**Last Updated**: April 2026  
**Status**: In Development
