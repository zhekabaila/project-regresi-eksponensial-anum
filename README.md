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

## 🚀 Cara Menjalankan Project

### **Step 1: Verifikasi Instalasi Python**

Pastikan Python 3.7 atau lebih tinggi sudah terinstall:

```bash
python3 --version
```

**Output yang diharapkan**: `Python 3.x.x` (minimal 3.7)

> ℹ️ Jika belum terinstall, download dari [python.org](https://www.python.org/downloads/)

---

### **Step 2: Clone/Setup Project**

```bash
# Navigasi ke folder project
cd /Users/zhekabaila/project-regresi-eksponensial-anum

# Atau setup dari awal jika belum ada
# git clone <repo-url>
# cd project-regresi-eksponensial-anum
```

---

### **Step 3: Buat Virtual Environment (Recommended)**

Virtual environment mengisolasi dependencies project Anda dari sistem Python global.

```bash
# Buat virtual environment dengan nama 'venv'
python3 -m venv venv

# Aktifkan virtual environment
source venv/bin/activate  # macOS/Linux
# atau untuk Windows:
# venv\Scripts\activate
```

**Verifikasi**: Terminal seharusnya menampilkan `(venv)` di depan prompt.

---

### **Step 4: Install Dependencies**

Install semua library yang diperlukan menggunakan `requirements.txt`:

```bash
# Pastikan Anda masih dalam virtual environment (venv aktif)
pip3 install -r requirements.txt
```

**Library yang akan diinstall**:

- `pandas` — manipulasi data
- `numpy` — komputasi numerik
- `scipy` — curve fitting & optimization
- `scikit-learn` — metrics (MAE, MSE, RMSE, R²)
- `matplotlib` — visualisasi grafik
- `seaborn` — statistical plots
- `jupyter` & `notebook` — Jupyter Notebook

**Proses instalasi** biasanya memakan waktu 1-3 menit tergantung kecepatan internet.

---

### **Step 5: Persiapkan Dataset**

Letakkan file `dataset.csv` di folder `data/`:

```bash
# Option 1: Copy file dari lokasi lain
cp /path/to/dataset.csv data/dataset.csv

# Option 2: Move file
mv /path/to/dataset.csv data/dataset.csv
```

**Verifikasi dataset tersimpan**:

```bash
ls -la data/dataset.csv
```

---

### **Step 6: Jalankan Jupyter Notebook**

Buka dan jalankan notebook analisis:

```bash
jupyter notebook notebooks/analisis_regresi.ipynb
```

**Output yang diharapkan**:

```
[I 14:30:45.123 NotebookApp] Serving notebooks from local directory: ...
[I 14:30:45.456 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/
```

Browser akan membuka otomatis. Jika tidak, buka [http://localhost:8888](http://localhost:8888) secara manual.

---

### **Step 7: Jalankan Cell Notebook Secara Berurutan**

Di Jupyter Notebook, jalankan setiap cell dari atas ke bawah:

1. **Cell 1**: Import libraries
2. **Cell 2**: Load dataset
3. **Cell 3-8**: Exploratory Data Analysis (EDA)
4. **Cell 9**: Preprocessing pipeline
5. **Cell 10-11**: Method 1 & Method 2 fitting
6. **Cell 12**: Evaluation & comparison
7. **Cell 13-15**: Visualisasi (akan generate gambar otomatis)
8. **Cell 16**: Parameter interpretation & critical point analysis

**Hasil output**:

- ✅ Konsol akan menampilkan nilai a, b, R², RMSE, MAE
- ✅ Gambar akan tersimpan otomatis di `output/figures/`
- ✅ Hasil model tersimpan di `output/hasil_model.txt`

---

### **Troubleshooting**

| Problem                                         | Solusi                                            |
| ----------------------------------------------- | ------------------------------------------------- |
| `ModuleNotFoundError: No module named 'pandas'` | Pastikan pip3 install berhasil: `pip3 list`       |
| Jupyter kernel stuck/error                      | Restart kernel: `Kernel → Restart & Clear Output` |
| Dataset tidak ditemukan                         | Periksa path: `ls data/dataset.csv`               |
| Virtual env tidak aktif                         | Jalankan: `source venv/bin/activate`              |
| Gambar tidak muncul di notebook                 | Tambah `%matplotlib inline` di cell pertama       |

---

### **Deaktivasi Virtual Environment (Opsional)**

Ketika selesai bekerja, deaktifkan virtual environment:

```bash
deactivate
```

---

### **Contoh Full Setup dari Awal (macOS)**

```bash
# 1. Cek Python
python3 --version

# 2. Masuk ke folder project
cd /Users/zhekabaila/project-regresi-eksponensial-anum

# 3. Buat & aktifkan venv
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip3 install -r requirements.txt

# 5. Copy dataset (ganti path sesuai lokasi Anda)
cp ~/Downloads/dataset.csv data/dataset.csv

# 6. Jalankan notebook
jupyter notebook notebooks/analisis_regresi.ipynb
```

✅ Selesai! Notebook siap digunakan.

## 📋 Tahapan Pengerjaan

- [x] Setup struktur folder
- [x] Setup environment & install dependencies
- [x] Create reusable Python modules (preprocessing, modeling, visualization)
- [x] Create Jupyter Notebook dengan 8 phases lengkap
- [ ] Load dan cleaning dataset
- [ ] Exploratory Data Analysis (EDA)
- [ ] Metode 1: Linearisasi + Least Squares
- [ ] Metode 2: SciPy Curve Fitting
- [ ] Perbandingan metode
- [ ] Evaluasi model (MAE, MSE, RMSE, R²)
- [ ] Visualisasi hasil (3 mandatory plots + optional)
- [ ] Penulisan laporan akademik (BAB I–V, ≥10 halaman)

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

- **Selalu filter** `focus_score > 0` sebelum logaritmik (critical rule!)
- **Gunakan kedua metode fitting**, bandingkan hasilnya
- **Dokumentasikan setiap keputusan** (outlier threshold, binning, filtering)
- **Simpan semua grafik** di `output/figures/` dengan DPI 150
- **Jalankan cell secara berurutan** di notebook (jangan skip atau acak)
- **Bersihkan output sebelum save**: `Kernel → Restart & Clear Output`

## 🎯 Output yang Diharapkan

Setelah menjalankan notebook, Anda akan mendapatkan:

### **1. Ringkasan Model** (`output/hasil_model.txt`)

```
focus_score = a · e^(b · social_media_hours)

a = [nilai hasil fitting]
b = [nilai hasil fitting]
R² = [nilai R²]
RMSE = [nilai RMSE]
```

### **2. Visualisasi Hasil** (`output/figures/`)

- `regresi_eksponensial.png` — Scatter plot + kurva eksponensial
- `residual_plot.png` — Residual analysis plot
- `heatmap_korelasi.png` — Correlation heatmap antar variabel
- `focus_by_bracket.png` — Bar chart focus per bracket durasi (optional)

### **3. Interpretasi Model**

- Nilai baseline fokus (parameter `a`)
- Laju penurunan fokus per jam (parameter `b`)
- Titik kritis (jam berapa fokus mencapai level berbahaya)

## 📄 Laporan Akhir (Deliverable Utama)

Setelah notebook berhasil dijalankan, laporan akademik harus mencakup:

### **Struktur BAB**

1. **BAB I: Pendahuluan** (±2 halaman)
   - Latar belakang masalah (pengaruh media sosial terhadap fokus belajar)
   - Rumusan masalah
   - Tujuan penelitian
   - Manfaat penelitian

2. **BAB II: Landasan Teori** (±2 halaman)
   - Fokus belajar mahasiswa (definisi, pentingnya)
   - Pengaruh media sosial terhadap pembelajaran
   - Teori regresi eksponensial (matematika, bentuk, transformasi)
   - Metrik evaluasi (R², RMSE, MAE)

3. **BAB III: Metodologi** (±2 halaman)
   - Dataset: sumber, jumlah baris, kolom yang digunakan
   - Preprocessing: handling missing value, filtering, outlier removal
   - Teknik pemodelan: Method 1 (linearisasi) & Method 2 (SciPy)
   - Tools & teknologi yang digunakan
   - **Catatan**: Sertakan output dari cell preprocessing (jumlah row sebelum/sesudah)

4. **BAB IV: Hasil & Pembahasan** (±3 halaman) ⭐ **PALING PENTING**
   - **4.1 Eksplorasi Data (EDA)**
     - Statistik deskriptif (min, max, mean, std dari X dan Y)
     - Korelasi antara social_media_hours dan focus_score
     - Masukkan screenshot `heatmap_korelasi.png` di sini
   - **4.2 Hasil Preprocessing**
     - Berapa banyak row yang dihapus dan mengapa
     - Range nilai setelah cleaning
   - **4.3 Persamaan Model**
     - Tampilkan persamaan final: `focus_score = a · e^(b · social_media_hours)`
     - Jelaskan nilai a dan b
     - Contoh interpretasi: "Ketika social_media_hours = 0, fokus = {a}"
   - **4.4 Hasil Evaluasi Model**
     - Tabel perbandingan Method 1 vs Method 2
   - **4.5 Visualisasi & Interpretasi**
     - Masukkan `regresi_eksponensial.png` dengan caption
     - Masukkan `residual_plot.png` dengan caption
     - Interpretasi hasil grafik
   - **4.6 Analisis Titik Kritis** (Optional)
     - Berapa jam media sosial saat fokus mencapai level 50?
     - Implikasi praktis dan rekomendasi

5. **BAB V: Penutup** (±1 halaman)
   - **5.1 Kesimpulan**: Ringkas temuan utama
   - **5.2 Saran**: Rekomendasi & penelitian lanjutan

---

### **Persyaratan Laporan**

- **Minimum**: 10 halaman (tidak termasuk cover & daftar isi)
- **Format**: A4, margin 1 inch, Times New Roman 12pt, spasi 1.5
- **File**: `laporan/laporan_akhir.docx` atau `.pdf`
- **Grafik**: Semua gambar dari `output/figures/` harus disertakan
- **Referensi**: Minimal 3 sumber akademik (format APA)
- **Lampiran**: Kode Python dari modul (preprocessing.py, modeling.py)

---

### **Tips Penulisan**

✅ **Yang harus dilakukan:**

- Sertakan semua grafik dari `output/figures/` dengan caption deskriptif
- Jelaskan setiap nilai konkret (a, b, R², RMSE) dengan narasi
- Gunakan bahasa Indonesia formal dan profesional
- Cantumkan sumber referensi untuk setiap klaim teoritis

❌ **Yang harus dihindari:**

- Jangan copy-paste kode ke bab utama (hanya di lampiran)
- Jangan lupa interpretasi hasil (selalu jelaskan "mengapa")
- Jangan menggunakan font/ukuran yang tidak konsisten
- Jangan submit tanpa proofreading

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Setup (satu kali saja)
cd /Users/zhekabaila/project-regresi-eksponensial-anum
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# 2. Copy dataset
cp ~/Downloads/dataset.csv data/dataset.csv

# 3. Jalankan notebook (everytime)
jupyter notebook notebooks/analisis_regresi.ipynb

# 4. Run all cells (Cell → Run All)
# Tunggu sampai selesai ~5-10 menit

# 5. Output tersedia di:
# - output/hasil_model.txt     (ringkasan model)
# - output/figures/*.png       (grafik hasil)

# 6. Tulis laporan BAB I–V
# - Masukkan nilai a, b, R² dari hasil
# - Sertakan gambar dari output/figures/
# - Simpan di laporan/laporan_akhir.docx

✅ DONE!
```

---

**Last Updated**: April 2026  
**Status**: Production Ready  
**Mata Kuliah**: Pemodelan & Simulasi
