"""
Visualization Module

Functions for plotting and visualizing exponential regression results,
including scatter plots, regression curves, residuals, and correlation heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import pandas as pd


def plot_regression_with_data(X: np.ndarray, Y: np.ndarray, Y_pred: np.ndarray, 
                               a: float, b: float, output_path: str = None,
                               title: str = "Exponential Regression") -> None:
    """
    Plot scatter data with fitted exponential curve overlay.
    
    Args:
        X (np.ndarray): Independent variable (social_media_hours)
        Y (np.ndarray): Actual dependent variable (focus_score)
        Y_pred (np.ndarray): Predicted values
        a (float): Parameter a
        b (float): Parameter b
        output_path (str): Path to save figure (optional)
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of actual data
    plt.scatter(X, Y, alpha=0.3, color='steelblue', label='Data Aktual', s=10)
    
    # Fitted curve
    X_line = np.linspace(X.min(), X.max(), 300)
    Y_line = a * np.exp(b * X_line)
    plt.plot(X_line, Y_line, color='red', linewidth=2.5, 
             label=f'Regresi Eksponensial: y = {a:.2f}·e^({b:.4f}x)')
    
    # Labels and formatting
    plt.xlabel('Durasi Penggunaan Media Sosial (jam/hari)', fontsize=12)
    plt.ylabel('Focus Score', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_residuals(Y: np.ndarray, Y_pred: np.ndarray, 
                   output_path: str = None, title: str = "Residual Plot") -> None:
    """
    Plot residuals vs predicted values to assess model fit quality.
    
    Args:
        Y (np.ndarray): Actual values
        Y_pred (np.ndarray): Predicted values
        output_path (str): Path to save figure (optional)
        title (str): Plot title
    """
    residuals = Y - Y_pred
    
    plt.figure(figsize=(10, 5))
    plt.scatter(Y_pred, residuals, alpha=0.3, color='darkorange', s=10)
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    
    plt.xlabel('Nilai Prediksi', fontsize=12)
    plt.ylabel('Residual (Aktual - Prediksi)', fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()
    
    # Print residual statistics
    print(f"\nResidual Statistics:")
    print(f"  Mean = {residuals.mean():.6f}")
    print(f"  Std  = {residuals.std():.6f}")
    print(f"  Min  = {residuals.min():.6f}")
    print(f"  Max  = {residuals.max():.6f}")


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str = None,
                              title: str = "Heatmap Korelasi Antar Variabel") -> None:
    """
    Plot correlation heatmap for all numeric variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_path (str): Path to save figure (optional)
        title (str): Plot title
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, cbar_kws={'label': 'Correlation'})
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()
    
    # Print main correlations with focus_score
    if 'focus_score' in numeric_df.columns:
        print(f"\nCorrelations with focus_score:")
        focus_corr = corr_matrix['focus_score'].sort_values(ascending=False)
        for var, corr_val in focus_corr.items():
            if var != 'focus_score':
                print(f"  {var:<25} = {corr_val:>7.4f}")


def plot_distribution_comparison(X: np.ndarray, Y: np.ndarray) -> None:
    """
    Plot histograms for both independent and dependent variables.
    
    Args:
        X (np.ndarray): Independent variable (social_media_hours)
        Y (np.ndarray): Dependent variable (focus_score)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # X distribution
    axes[0].hist(X, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Durasi Media Sosial (jam/hari)', fontsize=11)
    axes[0].set_ylabel('Frekuensi', fontsize=11)
    axes[0].set_title('Distribusi social_media_hours', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Y distribution
    axes[1].hist(Y, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Focus Score', fontsize=11)
    axes[1].set_ylabel('Frekuensi', fontsize=11)
    axes[1].set_title('Distribusi focus_score', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_focus_by_usage_brackets(df: pd.DataFrame, output_path: str = None) -> None:
    """
    Group data into media usage brackets and plot mean focus_score per bracket.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'social_media_hours' and 'focus_score'
        output_path (str): Path to save figure (optional)
    """
    # Create brackets
    brackets = [0, 2, 4, 6, 8, 10, 20]
    bracket_labels = ['0-2h', '2-4h', '4-6h', '6-8h', '8-10h', '10h+']
    
    df['usage_bracket'] = pd.cut(df['social_media_hours'], bins=brackets, 
                                  labels=bracket_labels, right=False)
    
    # Calculate mean focus_score per bracket
    bracket_means = df.groupby('usage_bracket', observed=True)['focus_score'].agg(['mean', 'std', 'count'])
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(bracket_means)), bracket_means['mean'], 
            color='steelblue', alpha=0.7, edgecolor='black')
    plt.errorbar(range(len(bracket_means)), bracket_means['mean'], 
                 yerr=bracket_means['std'], fmt='none', color='black', capsize=5)
    
    plt.xlabel('Durasi Penggunaan Media Sosial', fontsize=12)
    plt.ylabel('Rata-rata Focus Score', fontsize=12)
    plt.title('Focus Score per Bracket Durasi Media Sosial', fontsize=13)
    plt.xticks(range(len(bracket_means)), bracket_means.index, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()
    
    # Print table
    print(f"\nFocus Score Statistics by Usage Bracket:")
    print(bracket_means)
