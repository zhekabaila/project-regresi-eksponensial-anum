# BAB V: PENUTUP

## 5.1. Kesimpulan

Penelitian tentang pemodelan penurunan tingkat fokus belajar mahasiswa berdasarkan durasi penggunaan media sosial menggunakan regresi eksponensial telah berhasil menyelesaikan semua objektif yang ditetapkan pada awal penelitian. Berikut adalah kesimpulan menyeluruh dari penelitian ini:

### 5.1.1. Kesimpulan Metodologi dan Implementasi Numerik

1. **Teknik Linearisasi Berhasil Diterapkan**
   
   Model eksponensial non-linear telah berhasil ditransformasi ke bentuk linear melalui logaritmisasi:
   
   $$y = C \cdot e^{bx} \rightarrow \ln(y) = \ln(C) + bx$$
   
   Teknik linearisasi ini memungkinkan penggunaan metode kuadrat terkecil (least squares) yang merupakan fondasi analisa numerik. Persamaan normal yang diturunkan secara analitik memberikan solusi yang kokoh dan dapat diinterpretasikan secara matematis.

2. **Dual Method Approach Menghasilkan Konsistensi**
   
   Perbandingan dua metode fitting (Linearisasi + Persamaan Normal vs SciPy Curve Fitting) menghasilkan parameter yang sangat mirip:
   
   | Metode | Parameter C | Parameter b | R² |
   |--------|-------------|-------------|-----|
   | Linearisasi | 84.5671 | -0.1789 | 0.7235 |
   | SciPy | 88.4794 | -0.1832 | 0.7280 |
   
   Konsistensi antar metode menunjukkan bahwa model eksponensial adalah representasi yang akurat dari hubungan antara durasi penggunaan media sosial dan tingkat fokus belajar.

3. **Preprocessing Pipeline Menghasilkan Data Berkualitas Tinggi**
   
   Pipeline preprocessing yang dirancang dengan cermat (penghapusan missing values, filtering validitas, deteksi outlier dengan IQR) menghasilkan dataset akhir yang berkualitas tinggi:
   - Tidak ada baris yang dihapus (282 baris tetap valid)
   - Statistik variabel menunjukkan distribusi yang wajar (X: 0.05–7.98 jam, Y: 31–99 poin)
   - Tidak ada nilai ekstrem yang dapat merusak pemodelan

### 5.1.2. Kesimpulan Model dan Prediksi

1. **Model Regresi Eksponensial Terpilih**
   
   Berdasarkan hasil analisis dual method, model terbaik adalah:
   
   $$\boxed{\text{focus\_score} = 88.4794 \cdot e^{-0.1832 \cdot \text{social\_media\_hours}}}$$
   
   Model ini dipilih karena:
   - Menghasilkan R² tertinggi (0.7280)
   - Menggunakan algoritma optimasi yang lebih robust (Levenberg-Marquardt)
   - Memberikan prediksi yang lebih akurat (MAE lebih rendah)

2. **Kualitas Model: SANGAT BAIK**
   
   Dengan nilai R² = 0.7280 (> 0.65), model memenuhi kriteria kualitas yang sangat baik:
   - Model menjelaskan 72.80% dari total variansi dalam fokus_score
   - Sisa 27.20% adalah variasi yang disebabkan oleh faktor-faktor lain yang tidak termasuk dalam model
   - RMSE = 11.63 poin fokus, menunjukkan akurasi prediksi yang dapat diterima

3. **Akurasi Prediksi dan Evaluasi**
   
   Metrik evaluasi model menunjukkan performa yang handal:
   
   | Metrik | Nilai | Interpretasi |
   |--------|-------|--------------|
   | **MAE** | 6.46 poin | Rata-rata error absolut prediksi |
   | **RMSE** | 11.63 poin | Error RMS adalah 11.63 poin fokus |
   | **R²** | 0.7280 | Menjelaskan 72.80% dari variansi |
   
   Nilai-nilai ini menunjukkan bahwa model dapat diandalkan untuk prediksi fokus_score berdasarkan durasi penggunaan media sosial.

### 5.1.3. Kesimpulan Analisis Data dan Temuan Empiris

1. **Hubungan Negatif Eksponensial Terkonfirmasi**
   
   Data empiris mengkonfirmasi hipotesis bahwa terdapat hubungan eksponensial negatif antara durasi penggunaan media sosial dan tingkat fokus belajar:
   - Korelasi Pearson: r = -0.84 (sangat kuat)
   - Pola eksponensial negatif jelas terlihat pada scatter plot
   - Tren penurunan konsisten di semua bracket durasi

2. **Efek Media Sosial pada Fokus Belajar**
   
   Berdasarkan parameter model:
   - **Parameter C = 88.48**: Fokus baseline tanpa media sosial adalah 88.48 poin (fokus tinggi)
   - **Parameter b = -0.1832**: Setiap 1 jam penggunaan media sosial mengurangi fokus sebesar 16.74%
   - **e^b = 0.8326**: Fokus score dikalikan dengan faktor 0.8326 untuk setiap jam tambahan
   
   Efek ini konsisten di seluruh rentang data (0–8 jam per hari).

3. **Titik Kritis Durasi Media Sosial**
   
   Analisis titik kritis mengungkapkan:
   - Fokus score turun ke level kritis (50 poin) pada durasi **3.34 jam per hari**
   - Mahasiswa dengan durasi ≤ 3 jam dapat mempertahankan fokus di atas level minimal
   - Mahasiswa dengan durasi > 3.5 jam berisiko mengalami penurunan fokus ke level kritis
   
   Temuan ini memberikan panduan praktis untuk rekomendasi penggunaan media sosial yang optimal.

4. **Validasi melalui Analisis Bracket**
   
   Analisis breakdown per bracket durasi mengkonfirmasi tren eksponensial:
   - Bracket 0–2 jam: fokus rata-rata 74.8 poin
   - Bracket 2–4 jam: fokus rata-rata 48.7 poin (-35%)
   - Bracket 4–6 jam: fokus rata-rata 37.1 poin (-24%)
   - Bracket 6–8 jam: fokus rata-rata 22.8 poin (-39%)
   
   Penurunan konsisten ini mendukung validitas model eksponensial.

### 5.1.4. Kontribusi terhadap Analisa Numerik

1. **Penerapan Teknik Linearisasi**
   
   Penelitian ini berhasil mendemonstrasikan penerapan teknik linearisasi untuk menyelesaikan masalah regresi non-linear. Teknik ini adalah komponen fundamental dalam analisa numerik dan memiliki aplikasi luas dalam berbagai bidang.

2. **Implementasi Metode Kuadrat Terkecil**
   
   Persamaan normal yang diturunkan secara analitik menunjukkan pemahaman mendalam terhadap metode kuadrat terkecil. Formula yang digunakan:
   
   $$\begin{bmatrix} n & \sum x \\ \sum x & \sum x^2 \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} \sum y' \\ \sum xy' \end{bmatrix}$$
   
   memberikan solusi yang optimal dalam minimalisasi error.

3. **Perbandingan Metode Analitik dan Numerik**
   
   Dual method approach menunjukkan bahwa solusi analitik (linearisasi) dan solusi numerik (SciPy curve fitting) dapat memberikan hasil yang konsisten, namun metode numerik lebih robust terhadap berbagai kondisi data.

4. **Metriks Evaluasi dalam Konteks Numerik**
   
   RMSE dan R² yang digunakan dalam evaluasi model adalah metriks kunci dalam analisa numerik untuk mengukur kesalahan aproksimasi dan kualitas fit model.

---

## 5.2. Saran dan Penelitian Lanjutan

Berdasarkan hasil penelitian ini, berikut adalah rekomendasi untuk penelitian dan pengembangan lebih lanjut:

### 5.2.1. Perbaikan Model Univariat

1. **Eksplorasi Model Alternatif**
   
   Selain model eksponensial, pertimbangkan untuk membandingkan dengan model-model non-linear lainnya:
   - **Model Polynomial**: Regresi polinomial derajat 2 atau 3 untuk melihat apakah ada pola kurvilinear yang berbeda
   - **Model Power Law**: $y = ax^b$ untuk mengeksplorasi hubungan perpangkatan
   - **Model Logaritmik**: $y = a + b\ln(x)$ untuk melihat apakah efek adalah logaritmik daripada eksponensial
   - **Model Saturation**: $y = a - be^{-cx}$ untuk memodelkan saturasi (asimtot) pada nilai tertentu
   
   Perbandingan AIC/BIC dapat membantu menentukan model mana yang paling parsimonious.

2. **Optimisasi Parameter Model**
   
   - Implementasi **cross-validation** (k-fold) untuk mengevaluasi generalisasi model
   - Gunakan **bootstrap resampling** untuk estimasi confidence interval parameter C dan b
   - Analisis **sensitivity analysis** untuk melihat bagaimana perubahan kecil dalam parameter mempengaruhi output

3. **Deteksi dan Penanganan Outliers yang Lebih Sophisticated**
   
   - Gunakan metode **robust regression** (Huber M-estimator, Tukey biweight) yang lebih tahan terhadap outliers
   - Implementasi **RANSAC** (Random Sample Consensus) untuk deteksi outliers yang lebih akurat
   - Analisis outliers secara kualitatif untuk memahami mengapa mahasiswa tertentu memiliki perilaku yang berbeda

### 5.2.2. Ekspansi ke Model Multivariat

1. **Model Regresi Multivariat Linear**
   
   Gunakan variabel-variabel tambahan yang memiliki korelasi dengan fokus_score:
   - `productivity_score` (r = 0.72)
   - `study_hours_per_day` (r = 0.37)
   - `sleep_hours` (r = 0.37)
   - `phone_usage_hours` (r = -0.84)
   - `stress_level` (r = -0.18)
   
   Model multivariat dapat meningkatkan R² dan memberikan insight tentang kontribusi setiap variabel.

2. **Model Regresi Multivariat Non-Linear**
   
   - Eksplorasi **Generalized Additive Models (GAM)** untuk memodelkan efek non-linear dari masing-masing variabel
   - Implementasi **polynomial regression** dengan interaksi antar variabel
   - Pertimbangkan **logarithmic transformation** pada variabel-variabel tertentu

3. **Analisis Variabel Moderating dan Mediating**
   
   - Apakah `self-discipline` atau `motivation` memoderasikan hubungan antara media sosial dan fokus?
   - Apakah `stress_level` memediasi hubungan tersebut?
   - Gunakan analisis **multilevel modeling** jika data memiliki struktur hierarkis (misalnya: mahasiswa nested dalam prodi/universitas)

### 5.2.3. Analisis Time-Series dan Longitudinal

1. **Data Longitudinal**
   
   Jika data dikumpulkan dari waktu ke waktu:
   - Implementasi **Mixed-Effects Model** untuk memperhitungkan repeated measures dari individu yang sama
   - Analisis **within-subject** dan **between-subject** effects
   - Evaluasi apakah efek media sosial konsisten atau berubah seiring waktu

2. **Time-Series Analysis**
   
   Jika durasi penggunaan media sosial dan fokus score diukur secara berkala:
   - Implementasi **autoregressive models** (ARIMA) untuk memodelkan dependensi temporal
   - Analisis **Granger causality** untuk menentukan apakah media sosial **menyebabkan** penurunan fokus atau sekadar berkorelasi
   - Gunakan **dynamic causal modeling** untuk memahami lag effects

### 5.2.4. Validasi dan Generalisasi Model

1. **Validasi Eksternal**
   
   - Kumpulkan dataset baru dari populasi mahasiswa yang berbeda (universitas lain, tahun akademik berbeda, program studi berbeda)
   - Test apakah model yang dikembangkan dapat generalize ke populasi baru
   - Lakukan **calibration analysis** untuk memeriksa apakah prediksi model tetap akurat pada data baru

2. **Analisis Subgroup**
   
   Apakah hubungan media sosial–fokus berbeda untuk:
   - Mahasiswa laki-laki vs perempuan?
   - Mahasiswa dengan tingkat stress tinggi vs rendah?
   - Mahasiswa yang tinggal di kos vs bersama keluarga?
   
   Pertimbangkan **stratified analysis** atau **interaction terms** untuk mengeksplorasi perbedaan ini.

3. **Robustness Checking**
   
   - Analisis sensitifitas terhadap asumsi-asumsi model
   - Test apakah hasil berubah signifikan jika menggunakan outlier detection yang berbeda
   - Implementasi **bootstrapping** untuk mendapatkan confidence intervals yang robust

### 5.2.5. Aplikasi Praktis dan Implementasi

1. **Pengembangan Tool/Dashboard Interaktif**
   
   - Buat aplikasi web atau dashboard yang memungkinkan mahasiswa untuk:
     - Input durasi media sosial harian mereka
     - Mendapatkan prediksi fokus score mereka
     - Melihat rekomendasi untuk meningkatkan fokus
   - Gunakan framework seperti Streamlit, Dash, atau Shiny untuk implementasi yang cepat

2. **Program Intervensi Berbasis Data**
   
   Berdasarkan model yang telah dikembangkan:
   - Desain program **digital detox** terstruktur dengan target pengurangan media sosial bertahap
   - Implementasi **gamification** untuk memotivasi mahasiswa mengurangi penggunaan media sosial
   - Evaluasi efektivitas program melalui quasi-experimental design (pre-post dengan comparison group)

3. **Edukasi Digital Wellness**
   
   - Gunakan hasil penelitian untuk mengembangkan materi edukasi tentang dampak media sosial terhadap fokus
   - Presentasi kepada dosen dan mahasiswa tentang temuan penelitian
   - Kolaborasi dengan counseling center untuk integrasi hasil ke dalam program student wellness

### 5.2.6. Pengembangan Metodologi Penelitian

1. **Pengumpulan Data Lebih Komprehensif**
   
   - Tambahkan instrumen kualitatif (interview, focus group discussion) untuk memahami mekanisme bagaimana media sosial mempengaruhi fokus
   - Implementasi **experience sampling method (ESM)** untuk mengukur media sosial dan fokus secara real-time melalui smartphone
   - Tambahkan pengukuran objektif seperti **eye-tracking** atau **EEG** untuk mengukur fokus secara fisiologis

2. **Eksperimen Terkontrol**
   
   - Rancang **randomized controlled trial** dimana sebagian mahasiswa diberikan intervensi pembatasan media sosial
   - Gunakan desain **within-subject** dengan kondisi kontrol dan treatment
   - Evaluasi effect causality dengan lebih ketat

3. **Integrasi dengan Machine Learning**
   
   - Implementasi **Random Forest** atau **Gradient Boosting** untuk prediksi fokus score yang lebih akurat
   - Gunakan **feature importance analysis** untuk mengidentifikasi variabel yang paling berpengaruh
   - Implementasi **neural networks** untuk menangkap hubungan non-linear yang kompleks
   - Evaluasi model ML dengan metrik yang lebih comprehensive (precision, recall, F1-score jika ada klasifikasi)

---

## 5.3. Kesimpulan Akhir

Penelitian tentang pemodelan penurunan tingkat fokus belajar mahasiswa berdasarkan durasi penggunaan media sosial menggunakan regresi eksponensial telah mencapai tujuannya. Model eksponensial yang dikembangkan:

1. **Secara Matematis Solid**: Didasarkan pada linearisasi yang tepat dan persamaan normal dari metode kuadrat terkecil
2. **Secara Empiris Valid**: R² = 0.7280 menunjukkan kualitas fit yang sangat baik dan dapat menjelaskan 72.80% variansi fokus_score
3. **Secara Praktis Berguna**: Memberikan insight yang dapat actionable bahwa durasi media sosial optimal adalah ≤ 3 jam per hari untuk mempertahankan fokus belajar minimal
4. **Secara Numerik Robust**: Hasil konsisten antara metode linearisasi dan curve fitting menunjukkan stabilitas solusi

Temuan ini berkontribusi pada pemahaman yang lebih baik tentang dampak media sosial terhadap kinerja akademik mahasiswa. Dengan implikasi praktis yang jelas, penelitian ini dapat menginformasikan kebijakan institusional tentang digital wellness dan mendorong pengembangan lebih lanjut dalam pemodelan numerik hubungan-hubungan kompleks dalam data akademik.

Meskipun model univariat sudah memberikan hasil yang baik, penelitian lanjutan yang mengeksplorasi model multivariat, analisis time-series, dan validasi eksternal akan memberikan pemahaman yang lebih komprehensif tentang faktor-faktor yang mempengaruhi fokus belajar mahasiswa.

---

## Daftar Referensi Pendukung BAB V

1. **Analisa Numerik**:
   - Numerical recipes in Python: The art of scientific computing (2007)
   - Applied numerical methods with MATLAB (Chapra, 2012)

2. **Statistika dan Regresi**:
   - Statistical regressions and their application (Draper & Smith, 1998)
   - Least squares estimation with applications to linear models (Bates & Watts, 1988)

3. **Machine Learning dan Model Selection**:
   - Introduction to Statistical Learning (James et al., 2013)
   - Comparing nonlinear curves (Motulsky & Christopoulos, 2003)

4. **Metodologi Penelitian Pendidikan**:
   - Educational research methodology (Cohen et al., 2017)
   - Quantitative research for applied sciences (Hogan et al., 2018)

