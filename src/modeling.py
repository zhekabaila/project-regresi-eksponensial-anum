"""
Modul Pemodelan Regresi Eksponensial — Analisa Numerik

Implementasi DUA metode fitting untuk regresi eksponensial:
1. Linearisasi + Kuadrat Terkecil (Metode 1, inti Analisa Numerik)
2. SciPy Curve Fitting (Metode 2, pembanding)

Bentuk model: y = C * e^(b * x)
Dengan C dan b dihitung melalui persamaan normal (least squares method).
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict
import json


def fungsi_eksponensial(x: np.ndarray, C: float, b: float) -> np.ndarray:
    """
    Fungsi eksponensial: y = C * e^(b * x)
    
    Args:
        x (np.ndarray): Variabel independen
        C (float): Parameter amplitudo (konstanta awal)
        b (float): Parameter laju penurunan/pertumbuhan
        
    Returns:
        np.ndarray: Nilai prediksi
    """
    return C * np.exp(b * x)


def metode1_linearisasi(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    METODE 1: Linearisasi + Persamaan Normal (Inti Analisa Numerik)
    
    Transformasi: y = C*e^(bx) → ln(y) = ln(C) + b*x
    Kemudian gunakan polyfit(x, ln(y), 1) untuk mendapatkan koefisien regresi linear.
    
    Komponen persamaan normal yang dihitung:
    - n: jumlah data
    - Σx: jumlah nilai x
    - ΣY': jumlah nilai ln(Y)
    - Σx²: jumlah nilai x kuadrat
    - ΣxY': jumlah nilai x*ln(Y)
    
    Args:
        X (np.ndarray): Variabel independen (social_media_hours)
        Y (np.ndarray): Variabel dependen (focus_score)
        
    Returns:
        Tuple[float, float, np.ndarray]:
            - C: Parameter amplitudo
            - b: Parameter laju penurunan
            - Y_pred: Nilai prediksi
    """
    print("\n" + "=" * 60)
    print("METODE 1: LINEARISASI + PERSAMAAN NORMAL")
    print("=" * 60)
    
    # Transformasi ke ruang linear
    Y_prime = np.log(Y)  # Y' = ln(Y)
    
    # Regresi linear: ln(y) = ln(C) + b*x
    koefisien = np.polyfit(X, Y_prime, 1)  # Returns [b, ln(C)]
    b = koefisien[0]
    ln_C = koefisien[1]
    C = np.exp(ln_C)
    
    # Hitung komponen persamaan normal (untuk dokumentasi)
    n = len(X)
    sum_x = np.sum(X)
    sum_Yp = np.sum(Y_prime)
    sum_x2 = np.sum(X**2)
    sum_xYp = np.sum(X * Y_prime)
    
    # Prediksi
    Y_pred = fungsi_eksponensial(X, C, b)
    
    print(f"Persamaan: y = {C:.4f} * e^({b:.6f} * x)")
    print(f"  Parameter C = {C:.4f}")
    print(f"  Parameter b = {b:.6f}")
    print(f"\nKomponen Persamaan Normal:")
    print(f"  n = {n}")
    print(f"  Σx = {sum_x:.2f}")
    print(f"  ΣY' = {sum_Yp:.2f}")
    print(f"  Σx² = {sum_x2:.2f}")
    print(f"  ΣxY' = {sum_xYp:.2f}")
    
    return C, b, Y_pred


def metode2_scipy_curve_fit(X: np.ndarray, Y: np.ndarray, 
                             p0: list = None, maxfev: int = 10000) -> Tuple[float, float, np.ndarray]:
    """
    METODE 2: SciPy Curve Fitting (Algoritma Levenberg-Marquardt)
    
    Optimisasi yang lebih robust menggunakan scipy.optimize.curve_fit.
    Digunakan sebagai pembanding terhadap Metode 1.
    
    Args:
        X (np.ndarray): Variabel independen
        Y (np.ndarray): Variabel dependen
        p0 (list): Tebakan awal [C, b]. Default: [max(Y), -0.1]
        maxfev (int): Maksimal evaluasi fungsi (default: 10000)
        
    Returns:
        Tuple[float, float, np.ndarray]:
            - C: Parameter amplitudo yang sudah optimal
            - b: Parameter laju penurunan yang sudah optimal
            - Y_pred: Nilai prediksi
    """
    print("\n" + "=" * 60)
    print("METODE 2: SCIPY CURVE FITTING (LEVENBERG-MARQUARDT)")
    print("=" * 60)
    
    # Tebakan awal default
    if p0 is None:
        p0 = [max(Y), -0.1]
    
    try:
        # Curve fitting
        popt, pcov = curve_fit(fungsi_eksponensial, X, Y, 
                               p0=p0, maxfev=maxfev)
        C_opt, b_opt = popt
        
        # Prediksi
        Y_pred = fungsi_eksponensial(X, C_opt, b_opt)
        
        print(f"Persamaan: y = {C_opt:.4f} * e^({b_opt:.6f} * x)")
        print(f"  Parameter C = {C_opt:.4f}")
        print(f"  Parameter b = {b_opt:.6f}")
        print(f"Konvergensi: Berhasil")
        
        return C_opt, b_opt, Y_pred
    
    except RuntimeError as e:
        print(f"Konvergensi GAGAL: {e}")
        print(f"Mencoba dengan maxfev={maxfev*5}...")
        return metode2_scipy_curve_fit(X, Y, p0=p0, maxfev=maxfev*5)


def evaluasi_model(Y_aktual: np.ndarray, Y_prediksi: np.ndarray, 
                   nama_model: str = "") -> Dict[str, float]:
    """
    Evaluasi model menggunakan beberapa metrik.
    
    Metrik yang dihitung:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error (Galat RMS) — UTAMA untuk Analisa Numerik
    - R²: Koefisien Determinasi
    
    Args:
        Y_aktual (np.ndarray): Nilai aktual
        Y_prediksi (np.ndarray): Nilai prediksi
        nama_model (str): Nama model (untuk display)
        
    Returns:
        Dict[str, float]: Dictionary metrik evaluasi
    """
    mae = mean_absolute_error(Y_aktual, Y_prediksi)
    mse = mean_squared_error(Y_aktual, Y_prediksi)
    galat_RMS = np.sqrt(mse)  # RMSE = Galat RMS
    r2 = r2_score(Y_aktual, Y_prediksi)
    
    metrik = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': galat_RMS,
        'R2': r2
    }
    
    print(f"\n{nama_model}")
    print(f"  MAE  = {mae:.6f}")
    print(f"  MSE  = {mse:.6f}")
    print(f"  RMSE (Galat RMS) = {galat_RMS:.6f}")
    print(f"  R²   = {r2:.6f}")
    
    # Penilaian
    if r2 >= 0.65:
        print(f"  ✓ R² sangat baik (≥ 0.65)")
    elif r2 >= 0.40:
        print(f"  ✓ R² dapat diterima (≥ 0.40)")
    else:
        print(f"  ⚠ R² di bawah threshold (< 0.40) — pertimbangkan penyempurnaan model")
    
    return metrik


def bandingkan_metode(X: np.ndarray, Y: np.ndarray) -> Dict:
    """
    Jalankan KEDUA metode fitting dan bandingkan hasil.
    
    Args:
        X (np.ndarray): Variabel independen
        Y (np.ndarray): Variabel dependen
        
    Returns:
        Dict: Hasil perbandingan dengan semua parameter dan metrik
    """
    print("\n" + "=" * 70)
    print("FITTING REGRESI EKSPONENSIAL: PENDEKATAN DUAL METHOD")
    print("=" * 70)
    
    # Metode 1
    C1, b1, Y_pred1 = metode1_linearisasi(X, Y)
    metrik1 = evaluasi_model(Y, Y_pred1, nama_model="Metrik Metode 1")
    
    # Metode 2
    C2, b2, Y_pred2 = metode2_scipy_curve_fit(X, Y)
    metrik2 = evaluasi_model(Y, Y_pred2, nama_model="Metrik Metode 2")
    
    # Tabel perbandingan
    print("\n" + "=" * 75)
    print("TABEL PERBANDINGAN METODE")
    print("=" * 75)
    print(f"{'Metrik':<15} {'Metode 1 (Linearisasi)':<30} {'Metode 2 (SciPy)':<30}")
    print("-" * 75)
    print(f"{'Parameter C':<15} {C1:<30.6f} {C2:<30.6f}")
    print(f"{'Parameter b':<15} {b1:<30.6f} {b2:<30.6f}")
    print(f"{'MAE':<15} {metrik1['MAE']:<30.6f} {metrik2['MAE']:<30.6f}")
    print(f"{'RMSE':<15} {metrik1['RMSE']:<30.6f} {metrik2['RMSE']:<30.6f}")
    print(f"{'R²':<15} {metrik1['R2']:<30.6f} {metrik2['R2']:<30.6f}")
    print("=" * 75)
    
    # Tentukan metode terbaik
    metode_terbaik = "Metode 2 (SciPy)" if metrik2['R2'] >= metrik1['R2'] else "Metode 1"
    print(f"\n✓ DIREKOMENDASIKAN: {metode_terbaik}")
    
    return {
        'metode1': {
            'C': C1, 'b': b1, 'Y_pred': Y_pred1,
            'metrik': metrik1
        },
        'metode2': {
            'C': C2, 'b': b2, 'Y_pred': Y_pred2,
            'metrik': metrik2
        }
    }


def hitung_titik_kritis(C: float, b: float, threshold: float = 50) -> float:
    """
    Hitung jam media sosial kritis dimana focus_score mencapai threshold.
    
    Selesaikan: threshold = C * e^(b * x) untuk x
    x = ln(threshold / C) / b
    
    Args:
        C (float): Parameter amplitudo
        b (float): Parameter laju penurunan
        threshold (float): Nilai target focus_score (default: 50)
        
    Returns:
        float: Nilai x kritis (social_media_hours)
    """
    x_kritis = np.log(threshold / C) / b
    return x_kritis


def simpan_hasil_model(filepath: str, C: float, b: float, 
                       metrik: Dict[str, float], jumlah_data: int):
    """
    Simpan hasil model ke file teks.
    
    Args:
        filepath (str): Path file output
        C (float): Parameter C
        b (float): Parameter b
        metrik (Dict): Metrik evaluasi
        jumlah_data (int): Jumlah sampel yang digunakan
    """
    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("HASIL MODEL REGRESI EKSPONENSIAL\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PERSAMAAN MODEL\n")
        f.write("-" * 60 + "\n")
        f.write(f"focus_score = {C:.6f} * e^({b:.6f} * social_media_hours)\n\n")
        
        f.write("PARAMETER\n")
        f.write("-" * 60 + "\n")
        f.write(f"C = {C:.6f}\n")
        f.write(f"b = {b:.6f}\n")
        f.write(f"Jumlah data = {jumlah_data}\n\n")
        
        f.write("METRIK EVALUASI\n")
        f.write("-" * 60 + "\n")
        for nama_metrik, nilai_metrik in metrik.items():
            f.write(f"{nama_metrik:<8} = {nilai_metrik:.6f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Dihasilkan: April 2026\n")
    
    print(f"\nHasil disimpan ke {filepath}")
