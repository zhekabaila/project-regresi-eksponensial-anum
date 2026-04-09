"""
Exponential Regression Modeling Module

Implements TWO fitting methods for exponential regression:
1. Linearization + Least Squares (Method 1)
2. SciPy Curve Fitting (Method 2, recommended)

Model form: y = a * e^(b * x)
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict
import json


def exponential_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Exponential function: y = a * e^(b * x)
    
    Args:
        x (np.ndarray): Independent variable
        a (float): Amplitude parameter
        b (float): Decay/growth rate parameter
        
    Returns:
        np.ndarray: Predicted values
    """
    return a * np.exp(b * x)


def method1_linearization(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    METHOD 1: Linearization + Least Squares
    
    Transform: y = a*e^(bx) → ln(y) = ln(a) + b*x
    Then apply polyfit(x, ln(y), 1) to get linear regression coefficients.
    
    Args:
        X (np.ndarray): Independent variable (social_media_hours)
        Y (np.ndarray): Dependent variable (focus_score)
        
    Returns:
        Tuple[float, float, np.ndarray]:
            - a: Amplitude parameter
            - b: Decay rate parameter
            - Y_pred: Predicted values
    """
    print("\n" + "=" * 60)
    print("METHOD 1: LINEARIZATION + LEAST SQUARES")
    print("=" * 60)
    
    # Transform to linear space
    ln_Y = np.log(Y)
    
    # Linear regression: ln(y) = ln(a) + b*x
    coeffs = np.polyfit(X, ln_Y, 1)  # Returns [b, ln(a)]
    b = coeffs[0]
    ln_a = coeffs[1]
    a = np.exp(ln_a)
    
    # Predict
    Y_pred = exponential_function(X, a, b)
    
    print(f"Persamaan: y = {a:.4f} * e^({b:.6f} * x)")
    print(f"  a = {a:.4f}")
    print(f"  b = {b:.6f}")
    
    return a, b, Y_pred


def method2_scipy_curve_fit(X: np.ndarray, Y: np.ndarray, 
                             p0: list = None, maxfev: int = 10000) -> Tuple[float, float, np.ndarray]:
    """
    METHOD 2: SciPy Curve Fitting (Levenberg-Marquardt Algorithm)
    
    More robust optimization using scipy.optimize.curve_fit.
    Recommended for better convergence and handling edge cases.
    
    Args:
        X (np.ndarray): Independent variable
        Y (np.ndarray): Dependent variable
        p0 (list): Initial guess [a, b]. Default: [max(Y), -0.1]
        maxfev (int): Maximum function evaluations (default: 10000)
        
    Returns:
        Tuple[float, float, np.ndarray]:
            - a: Optimized amplitude parameter
            - b: Optimized decay rate parameter
            - Y_pred: Predicted values
    """
    print("\n" + "=" * 60)
    print("METHOD 2: SCIPY CURVE FITTING (LEVENBERG-MARQUARDT)")
    print("=" * 60)
    
    # Default initial guess
    if p0 is None:
        p0 = [max(Y), -0.1]
    
    try:
        # Curve fitting
        popt, pcov = curve_fit(exponential_function, X, Y, 
                               p0=p0, maxfev=maxfev)
        a_opt, b_opt = popt
        
        # Predict
        Y_pred = exponential_function(X, a_opt, b_opt)
        
        print(f"Persamaan: y = {a_opt:.4f} * e^({b_opt:.6f} * x)")
        print(f"  a = {a_opt:.4f}")
        print(f"  b = {b_opt:.6f}")
        print(f"Convergence: Success")
        
        return a_opt, b_opt, Y_pred
    
    except RuntimeError as e:
        print(f"Convergence FAILED: {e}")
        print(f"Trying with larger maxfev={maxfev*5}...")
        return method2_scipy_curve_fit(X, Y, p0=p0, maxfev=maxfev*5)


def evaluate_model(Y_true: np.ndarray, Y_pred: np.ndarray, 
                   model_name: str = "") -> Dict[str, float]:
    """
    Evaluate model using multiple metrics.
    
    Metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - R²: Coefficient of Determination
    
    Args:
        Y_true (np.ndarray): Actual values
        Y_pred (np.ndarray): Predicted values
        model_name (str): Name of the model (for display)
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    mae = mean_absolute_error(Y_true, Y_pred)
    mse = mean_squared_error(Y_true, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_true, Y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }
    
    print(f"\n{model_name}")
    print(f"  MAE  = {mae:.6f}")
    print(f"  MSE  = {mse:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  R²   = {r2:.6f}")
    
    # Assessment
    if r2 >= 0.65:
        print(f"  ✓ R² is excellent (≥ 0.65)")
    elif r2 >= 0.40:
        print(f"  ✓ R² is acceptable (≥ 0.40)")
    else:
        print(f"  ⚠ R² is below threshold (< 0.40) — consider model refinement")
    
    return metrics


def compare_methods(X: np.ndarray, Y: np.ndarray) -> Dict:
    """
    Run BOTH fitting methods and compare results.
    
    Args:
        X (np.ndarray): Independent variable
        Y (np.ndarray): Dependent variable
        
    Returns:
        Dict: Comparison results with all parameters and metrics
    """
    print("\n" + "=" * 70)
    print("FITTING EXPONENTIAL REGRESSION: DUAL METHOD APPROACH")
    print("=" * 70)
    
    # Method 1
    a1, b1, Y_pred1 = method1_linearization(X, Y)
    metrics1 = evaluate_model(Y, Y_pred1, model_name="Method 1 Metrics")
    
    # Method 2
    a2, b2, Y_pred2 = method2_scipy_curve_fit(X, Y)
    metrics2 = evaluate_model(Y, Y_pred2, model_name="Method 2 Metrics")
    
    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<15} {'Method 1 (Linearization)':<30} {'Method 2 (SciPy)':<30}")
    print("-" * 75)
    print(f"{'a':<15} {a1:<30.6f} {a2:<30.6f}")
    print(f"{'b':<15} {b1:<30.6f} {b2:<30.6f}")
    print(f"{'MAE':<15} {metrics1['MAE']:<30.6f} {metrics2['MAE']:<30.6f}")
    print(f"{'RMSE':<15} {metrics1['RMSE']:<30.6f} {metrics2['RMSE']:<30.6f}")
    print(f"{'R²':<15} {metrics1['R²']:<30.6f} {metrics2['R²']:<30.6f}")
    print("=" * 75)
    
    # Determine best method
    best_method = "Method 2 (SciPy)" if metrics2['R²'] >= metrics1['R²'] else "Method 1"
    print(f"\n✓ RECOMMENDED: {best_method}")
    
    return {
        'method1': {
            'a': a1, 'b': b1, 'Y_pred': Y_pred1,
            'metrics': metrics1
        },
        'method2': {
            'a': a2, 'b': b2, 'Y_pred': Y_pred2,
            'metrics': metrics2
        }
    }


def calculate_critical_point(a: float, b: float, threshold: float = 50) -> float:
    """
    Calculate critical social_media_hours where focus_score reaches threshold.
    
    Solve: threshold = a * e^(b * x) for x
    x = ln(threshold / a) / b
    
    Args:
        a (float): Amplitude parameter
        b (float): Decay rate parameter
        threshold (float): Target focus_score value (default: 50)
        
    Returns:
        float: Critical x value (social_media_hours)
    """
    x_critical = np.log(threshold / a) / b
    return x_critical


def save_model_results(filepath: str, a: float, b: float, 
                       metrics: Dict[str, float], dataset_size: int):
    """
    Save model results to text file.
    
    Args:
        filepath (str): Output file path
        a (float): Parameter a
        b (float): Parameter b
        metrics (Dict): Evaluation metrics
        dataset_size (int): Number of samples used
    """
    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPONENTIAL REGRESSION MODEL RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MODEL EQUATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"focus_score = {a:.6f} * e^({b:.6f} * social_media_hours)\n\n")
        
        f.write("PARAMETERS\n")
        f.write("-" * 60 + "\n")
        f.write(f"a = {a:.6f}\n")
        f.write(f"b = {b:.6f}\n")
        f.write(f"Dataset size = {dataset_size}\n\n")
        
        f.write("EVALUATION METRICS\n")
        f.write("-" * 60 + "\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name:<8} = {metric_value:.6f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Generated: April 2026\n")
    
    print(f"\nResults saved to {filepath}")
