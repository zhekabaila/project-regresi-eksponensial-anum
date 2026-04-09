"""
Modul Preprocessing Data — Analisa Numerik

Fungsi untuk membersihkan, validasi, dan persiapan data untuk pemodelan regresi eksponensial.
Menangani missing values, filtering, deteksi outlier, dan transformasi data.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_and_validate_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset CSV dan lakukan validasi awal.
    
    Args:
        filepath (str): Path ke file CSV dataset
        
    Returns:
        pd.DataFrame: Dataset yang telah di-load
        
    Raises:
        FileNotFoundError: Jika file tidak ditemukan
        ValueError: Jika kolom yang diperlukan tidak ada
    """
    df = pd.read_csv(filepath)
    
    # Cek kolom yang diperlukan
    required_cols = ['social_media_hours', 'focus_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom yang hilang: {missing_cols}")
    
    print(f"Dataset berhasil di-load: {df.shape[0]} baris × {df.shape[1]} kolom")
    return df


def remove_missing_values(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Hapus baris dengan missing values di kolom yang ditentukan.
    
    Args:
        df (pd.DataFrame): Input dataframe
        subset (list): Kolom yang dicek untuk missing values.
                      Default: ['social_media_hours', 'focus_score']
        
    Returns:
        pd.DataFrame: Dataframe yang telah dibersihkan
    """
    if subset is None:
        subset = ['social_media_hours', 'focus_score']
    
    rows_before = len(df)
    df_clean = df.dropna(subset=subset)
    rows_removed = rows_before - len(df_clean)
    
    print(f"Missing values: {rows_removed} baris dihapus ({rows_removed/rows_before*100:.2f}%)")
    return df_clean


def filter_valid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset untuk menyimpan hanya nilai yang valid untuk regresi eksponensial.
    
    ATURAN KRITIS (WAJIB untuk Analisa Numerik):
    - focus_score > 0 (diperlukan untuk transformasi logaritma)
    - social_media_hours >= 0 (tidak boleh negatif)
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe yang difilter dengan nilai valid saja
    """
    rows_before = len(df)
    
    # Filter focus_score > 0 (CRITICAL untuk ln())
    df_valid = df[df['focus_score'] > 0].copy()
    
    # Filter social_media_hours >= 0
    df_valid = df_valid[df_valid['social_media_hours'] >= 0].copy()
    
    rows_removed = rows_before - len(df_valid)
    print(f"Filter validitas: {rows_removed} baris dihapus karena nilai tidak valid ({rows_removed/rows_before*100:.2f}%)")
    
    return df_valid


def detect_outliers_iqr(df: pd.DataFrame, columns: list = None, 
                        iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Deteksi dan hapus outlier menggunakan metode Interquartile Range (IQR).
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Kolom untuk deteksi outlier.
                       Default: ['social_media_hours', 'focus_score']
        iqr_multiplier (float): Pengali IQR untuk threshold (default: 1.5)
        
    Returns:
        pd.DataFrame: Dataframe dengan outlier telah dihapus
    """
    if columns is None:
        columns = ['social_media_hours', 'focus_score']
    
    df_clean = df.copy()
    rows_before = len(df_clean)
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        print(f"Kolom '{col}': {outliers} outlier dihapus (batas: [{lower_bound:.2f}, {upper_bound:.2f}])")
    
    rows_removed = rows_before - len(df_clean)
    print(f"Total outlier dihapus: {rows_removed} baris ({rows_removed/rows_before*100:.2f}%)")
    
    return df_clean


def preprocess_pipeline(filepath: str, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Pipeline preprocessing lengkap: load → validasi → filter → hapus outlier.
    
    Args:
        filepath (str): Path ke file CSV
        verbose (bool): Tampilkan pesan progress
        
    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]: 
            - X: social_media_hours (numpy array)
            - Y: focus_score (numpy array)
            - df_clean: Dataframe yang telah dibersihkan
    """
    print("=" * 60)
    print("PIPELINE PREPROCESSING")
    print("=" * 60)
    
    # Langkah 1: Load
    df = load_and_validate_dataset(filepath)
    
    # Langkah 2: Hapus missing values
    df = remove_missing_values(df)
    
    # Langkah 3: Filter nilai valid (CRITICAL)
    df = filter_valid_values(df)
    
    # Langkah 4: Hapus outlier
    df = detect_outliers_iqr(df)
    
    # Ekstrak X dan Y
    X = df['social_media_hours'].values
    Y = df['focus_score'].values
    
    print(f"\nDataset final: {len(X)} baris")
    print(f"X (social_media_hours): min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
    print(f"Y (focus_score): min={Y.min():.2f}, max={Y.max():.2f}, mean={Y.mean():.2f}")
    print("=" * 60)
    
    return X, Y, df


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dapatkan statistik deskriptif untuk dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Statistik ringkas
    """
    return df.describe()
