# Task 1: Dataset Exploration (Diabetes dataset)

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 1) Loading the dataset
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# 2) Exploring the dataset
print("Dataset Overview")
print("Feature names:", feature_names)
print("X shape:", X.shape)  # (n_samples, n_features)
print("y shape:", y.shape)  # (n_samples,)
print("Target (y) summary:")
print(f"  min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}\n")

# Put features into a DataFrame for easy viewing/stats
df_X = pd.DataFrame(X, columns=feature_names)
print("First 5 rows of features:")
print(df_X.head(), "\n")

print("Feature summary statistics:")
print(df_X.describe().T, "\n")


# 3) Splitting the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(" Train/Test Split ")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Task 2: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_full = r2_score(y_test, y_pred)

print(f"Test R^2 (all 10 features): {r2_full:.6f}\n")


# Task 3: Implementing RFE (manual elimination)

def rfe_linear_regression_path(X_train, y_train, X_test, y_test, feature_names):
    """
    RFE loop:
      - start with all features
      - fit LinearRegression
      - compute test R^2
      - remove the feature with smallest |coef|
      - repeat until 1 feature remains

    Returns:
      results_df: summary per iteration
      coef_table: coefficients per iteration (NaN for eliminated features)
      elimination_order: least->most important removed (last remaining is most important)
    """
    remaining = list(range(len(feature_names)))
    history = []
    elimination_order = []

    while True:
        model = LinearRegression()
        model.fit(X_train[:, remaining], y_train)

        r2 = r2_score(y_test, model.predict(X_test[:, remaining]))

        # store coefficients in a full-length vector (NaN for removed)
        coef_full = np.full(len(feature_names), np.nan, dtype=float)
        coef_full[remaining] = model.coef_

        # choose least important feature = smallest abs(coef) among remaining
        abs_coefs = np.abs(model.coef_)
        least_pos = int(np.argmin(abs_coefs))         
        least_global_idx = remaining[least_pos]       # index in original feature list
        least_name = feature_names[least_global_idx]

        history.append({
            "n_features": len(remaining),
            "r2": r2,
            "retained_features": [feature_names[i] for i in remaining],
            "eliminated_next": None if len(remaining) == 1 else least_name,
            "coef_full": coef_full
        })

        if len(remaining) == 1:
            break

        elimination_order.append(least_name)
        remaining.pop(least_pos)

    results_df = pd.DataFrame({
        "n_features": [h["n_features"] for h in history],
        "r2": [h["r2"] for h in history],
        "eliminated_next": [h["eliminated_next"] for h in history],
        "retained_features": [", ".join(h["retained_features"]) for h in history],
    })

    coef_table = pd.DataFrame(
        {h["n_features"]: h["coef_full"] for h in history},
        index=feature_names
    ).T  # rows = n_features, cols = features

    # sort so it reads from 10 -> 1
    results_df = results_df.sort_values("n_features", ascending=False).reset_index(drop=True)
    coef_table = coef_table.sort_index(ascending=False)

    return results_df, coef_table, elimination_order


results_df, coef_table, elimination_order = rfe_linear_regression_path(
    X_train, y_train, X_test, y_test, feature_names
)

print("Task 3")
print("R^2 at each iteration (10 -> 1 features):")
print(results_df[["n_features", "r2", "eliminated_next"]], "\n")

# 3.4 Visualize R^2 vs number of features
plt.figure()
plt.plot(results_df["n_features"], results_df["r2"], marker="o")
plt.xlabel("Number of retained features")
plt.ylabel("Test R^2")
plt.title("RFE with Linear Regression: R^2 vs #Features")
plt.gca().invert_xaxis()  # show 10 -> 1 left-to-right
plt.show()

# 3.5 Identify optimal number of features using a "significant improvement" threshold
threshold = 0.01
best_r2 = results_df["r2"].max()

# pick the SMALLEST feature count whose R^2 is within (threshold) of the best observed R^2
candidates = results_df[results_df["r2"] >= best_r2 - threshold].copy()
optimal_row = candidates.sort_values("n_features", ascending=True).iloc[0]
optimal_n = int(optimal_row["n_features"])
optimal_features = optimal_row["retained_features"].split(", ")

print(f"Best R^2 observed: {best_r2:.6f}")
print(f"Threshold: {threshold:.2f}")
print(f"Optimal #features (smallest within threshold of best): {optimal_n}")
print("Selected features:", optimal_features, "\n")


# Task 4: Analyze Feature Importance

# 4.1 Table of coefficients at each iteration
print("Coefficient table (rows = #features, cols = feature; NaN = eliminated):")
print(coef_table, "\n")

# also create a "long" version that is easy to paste into reports
coef_long = (
    coef_table.reset_index(names="n_features")
    .melt(id_vars="n_features", var_name="feature", value_name="coefficient")
    .sort_values(["n_features", "feature"], ascending=[False, True])
)
print("Long-format coefficient table (first 30 rows):")
print(coef_long.head(30), "\n")

# 4.2 Three most important features (by RFE ranking: last remaining are most important)
most_important = [f for f in feature_names if f not in elimination_order]  # final remaining (1 feature)
# build full importance order: most important -> least important
importance_order = most_important + elimination_order[::-1]
top3 = importance_order[:3]

print("RFE importance order (most -> least):")
print(importance_order)
print("Top 3 features:", top3, "\n")

# 4.3 Compare initial ranking (all-feature model) vs final selected features
full_model_ranking = sorted(
    zip(feature_names, np.abs(lr.coef_), lr.coef_),
    key=lambda t: t[1],
    reverse=True
)

print("Initial ranking from ALL-feature LinearRegression (by |coef|):")
for name, abs_c, c in full_model_ranking:
    print(f"  {name:>3s}: |coef|={abs_c:9.3f}, coef={c:9.3f}")

print("\nFinal selected feature set (optimal):")
print(optimal_features)