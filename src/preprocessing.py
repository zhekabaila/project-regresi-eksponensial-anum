"""
Data Preprocessing Module

Functions for cleaning, validating, and preparing data for exponential regression modeling.
Handles missing values, filtering, outlier detection, and transformation.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_and_validate_dataset(filepath: str) -> pd.DataFrame:
    """
    Load CSV dataset and perform initial validation.
    
    Args:
        filepath (str): Path to the CSV dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If required columns are missing
    """
    df = pd.read_csv(filepath)
    
    # Check required columns
    required_cols = ['social_media_hours', 'focus_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def remove_missing_values(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Remove rows with missing values in specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        subset (list): Columns to check for missing values.
                      Default: ['social_media_hours', 'focus_score']
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if subset is None:
        subset = ['social_media_hours', 'focus_score']
    
    rows_before = len(df)
    df_clean = df.dropna(subset=subset)
    rows_removed = rows_before - len(df_clean)
    
    print(f"Missing values: {rows_removed} rows removed ({rows_removed/rows_before*100:.2f}%)")
    return df_clean


def filter_valid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to keep only valid values for exponential regression.
    
    CRITICAL RULES:
    - focus_score > 0 (required for logarithmic transformation)
    - social_media_hours >= 0 (cannot be negative)
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Filtered dataframe with valid values only
    """
    rows_before = len(df)
    
    # Filter focus_score > 0
    df_valid = df[df['focus_score'] > 0].copy()
    
    # Filter social_media_hours >= 0
    df_valid = df_valid[df_valid['social_media_hours'] >= 0].copy()
    
    rows_removed = rows_before - len(df_valid)
    print(f"Validity filter: {rows_removed} rows removed for invalid values ({rows_removed/rows_before*100:.2f}%)")
    
    return df_valid


def detect_outliers_iqr(df: pd.DataFrame, columns: list = None, 
                        iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Detect and remove outliers using Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Columns to apply outlier detection.
                       Default: ['social_media_hours', 'focus_score']
        iqr_multiplier (float): IQR multiplier for threshold (default: 1.5)
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
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
        
        print(f"Column '{col}': {outliers} outliers removed (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
    
    rows_removed = rows_before - len(df_clean)
    print(f"Total outliers removed: {rows_removed} rows ({rows_removed/rows_before*100:.2f}%)")
    
    return df_clean


def preprocess_pipeline(filepath: str, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Complete preprocessing pipeline: load → validate → filter → remove outliers.
    
    Args:
        filepath (str): Path to CSV file
        verbose (bool): Print progress messages
        
    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]: 
            - X: social_media_hours (numpy array)
            - Y: focus_score (numpy array)
            - df_clean: Cleaned dataframe
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load
    df = load_and_validate_dataset(filepath)
    
    # Step 2: Remove missing
    df = remove_missing_values(df)
    
    # Step 3: Filter valid values (CRITICAL)
    df = filter_valid_values(df)
    
    # Step 4: Remove outliers
    df = detect_outliers_iqr(df)
    
    # Extract X and Y
    X = df['social_media_hours'].values
    Y = df['focus_score'].values
    
    print(f"\nFinal dataset: {len(X)} rows")
    print(f"X (social_media_hours): min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
    print(f"Y (focus_score): min={Y.min():.2f}, max={Y.max():.2f}, mean={Y.mean():.2f}")
    print("=" * 60)
    
    return X, Y, df


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get descriptive statistics for the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Summary statistics
    """
    return df.describe()
