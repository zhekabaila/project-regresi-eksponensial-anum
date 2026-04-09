"""
Modul Visualisasi — Analisa Numerik

Fungsi untuk plotting dan visualisasi hasil regresi eksponensial,
termasuk scatter plot, kurva regresi, residual, dan heatmap korelasi.
Semua label dan judul menggunakan Bahasa Indonesia.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import pandas as pd


def plot_regresi_dengan_data(X: np.ndarray, Y: np.ndarray, Y_prediksi: np.ndarray, 
                               C: float, b: float, output_path: str = None,
                               title: str = "Regresi Eksponensial") -> None:
    """
    Plot scatter data dengan overlay kurva eksponensial yang sudah fitting.
    
    Args:
        X (np.ndarray): Variabel independen (social_media_hours)
        Y (np.ndarray): Variabel dependen aktual (focus_score)
        Y_prediksi (np.ndarray): Nilai prediksi
        C (float): Parameter C
        b (float): Parameter b
        output_path (str): Path untuk menyimpan gambar (opsional)
        title (str): Judul plot
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot data aktual
    plt.scatter(X, Y, alpha=0.3, color='steelblue', label='Data Aktual', s=10)
    
    # Kurva fitting
    X_line = np.linspace(X.min(), X.max(), 300)
    Y_line = C * np.exp(b * X_line)
    plt.plot(X_line, Y_line, color='red', linewidth=2.5, 
             label=f'Kurva Prediksi: y = {C:.2f}·e^({b:.4f}x)')
    
    # Label dan formatting
    plt.xlabel('Durasi Penggunaan Media Sosial (jam/hari)', fontsize=12)
    plt.ylabel('Skor Fokus', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Simpan jika path diberikan
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Tersimpan: {output_path}")
    
    plt.show()


def plot_residual(Y: np.ndarray, Y_prediksi: np.ndarray, 
                   output_path: str = None, title: str = "Plot Residual") -> None:
    """
    Plot residual vs nilai prediksi untuk menilai kualitas fit model.
    
    Args:
        Y (np.ndarray): Nilai aktual
        Y_prediksi (np.ndarray): Nilai prediksi
        output_path (str): Path untuk menyimpan gambar (opsional)
        title (str): Judul plot
    """
    residual = Y - Y_prediksi
    
    plt.figure(figsize=(10, 5))
    plt.scatter(Y_prediksi, residual, alpha=0.3, color='darkorange', s=10)
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    
    plt.xlabel('Nilai Prediksi', fontsize=12)
    plt.ylabel('Residual (Aktual - Prediksi)', fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Simpan jika path diberikan
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Tersimpan: {output_path}")
    
    plt.show()
    
    # Tampilkan statistik residual
    print(f"\nStatistik Residual:")
    print(f"  Mean = {residual.mean():.6f}")
    print(f"  Std  = {residual.std():.6f}")
    print(f"  Min  = {residual.min():.6f}")
    print(f"  Max  = {residual.max():.6f}")


def plot_heatmap_korelasi(df: pd.DataFrame, output_path: str = None,
                              title: str = "Heatmap Korelasi Antar Variabel") -> None:
    """
    Plot heatmap korelasi untuk semua variabel numerik.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_path (str): Path untuk menyimpan gambar (opsional)
        title (str): Judul plot
    """
    # Pilih kolom numerik saja
    df_numerik = df.select_dtypes(include=[np.number])
    
    # Hitung matrik korelasi
    matrik_korelasi = df_numerik.corr()
    
    # Plot heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(matrik_korelasi, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, cbar_kws={'label': 'Korelasi'})
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    # Simpan jika path diberikan
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Tersimpan: {output_path}")
    
    plt.show()
    
    # Tampilkan korelasi utama dengan focus_score
    if 'focus_score' in df_numerik.columns:
        print(f"\nKorelasi dengan focus_score:")
        korelasi_fokus = matrik_korelasi['focus_score'].sort_values(ascending=False)
        for var, nilai_korelasi in korelasi_fokus.items():
            if var != 'focus_score':
                print(f"  {var:<25} = {nilai_korelasi:>7.4f}")


def plot_perbandingan_distribusi(X: np.ndarray, Y: np.ndarray) -> None:
    """
    Plot histogram untuk variabel independen dan dependen.
    
    Args:
        X (np.ndarray): Variabel independen (social_media_hours)
        Y (np.ndarray): Variabel dependen (focus_score)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Distribusi X
    axes[0].hist(X, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Durasi Media Sosial (jam/hari)', fontsize=11)
    axes[0].set_ylabel('Frekuensi', fontsize=11)
    axes[0].set_title('Distribusi social_media_hours', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Distribusi Y
    axes[1].hist(Y, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Skor Fokus', fontsize=11)
    axes[1].set_ylabel('Frekuensi', fontsize=11)
    axes[1].set_title('Distribusi focus_score', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_fokus_per_bracket(df: pd.DataFrame, output_path: str = None) -> None:
    """
    Kelompokkan data ke dalam bracket penggunaan media dan plot rata-rata focus_score per bracket.
    
    Args:
        df (pd.DataFrame): Input dataframe dengan 'social_media_hours' dan 'focus_score'
        output_path (str): Path untuk menyimpan gambar (opsional)
    """
    # Buat bracket
    bracket = [0, 2, 4, 6, 8, 10, 20]
    label_bracket = ['0-2j', '2-4j', '4-6j', '6-8j', '8-10j', '10j+']
    
    df['bracket_penggunaan'] = pd.cut(df['social_media_hours'], bins=bracket, 
                                      labels=label_bracket, right=False)
    
    # Hitung rata-rata focus_score per bracket
    rata_rata_bracket = df.groupby('bracket_penggunaan', observed=True)['focus_score'].agg(['mean', 'std', 'count'])
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(rata_rata_bracket)), rata_rata_bracket['mean'], 
            color='steelblue', alpha=0.7, edgecolor='black')
    plt.errorbar(range(len(rata_rata_bracket)), rata_rata_bracket['mean'], 
                 yerr=rata_rata_bracket['std'], fmt='none', color='black', capsize=5)
    
    plt.xlabel('Durasi Penggunaan Media Sosial', fontsize=12)
    plt.ylabel('Rata-rata Skor Fokus', fontsize=12)
    plt.title('Skor Fokus per Bracket Durasi Media Sosial', fontsize=13)
    plt.xticks(range(len(rata_rata_bracket)), rata_rata_bracket.index, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Simpan jika path diberikan
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Tersimpan: {output_path}")
    
    plt.show()
    
    # Tampilkan tabel
    print(f"\nStatistik Skor Fokus per Bracket Penggunaan:")
    print(rata_rata_bracket)
