# A2-RFE — Recursive Feature Elimination with Linear Regression (Diabetes Dataset)

This assignment demonstrates how to use **Recursive Feature Elimination (RFE)** with a **Linear Regression** model to identify important features in the **scikit-learn Diabetes dataset**, evaluate model performance, and analyze feature importance across iterations.

## Files in this repository

- `tasks.py` — Python script that completes Tasks 1–4:
  - Loads and explores the Diabetes dataset
  - Splits data into training/testing sets (80/20)
  - Trains a Linear Regression model and evaluates using **R²**
  - Performs RFE by iteratively removing the least important feature (smallest `|coef|`)
  - Tracks R² and coefficients at each iteration
  - Plots **R² vs number of retained features**
  - Prints tables/rankings for feature importance
- `A2_task5.pdf` — Task 5 reflection write-up 
- Few screenshots of the code working and a graph

## Requirements

- Python 3.9+ (tested with Python 3.13)
- Packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

## Setup (install dependencies)

### Option 1: Install with pip
```bash
pip install numpy pandas matplotlib scikit-learn