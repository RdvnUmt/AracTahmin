import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from utils import regression_metrics

# Load data
X_train = np.load("X_train_processed.npy")
X_test  = np.load("X_test_processed.npy")
y_train = np.load("y_train.npy").reshape(-1)
y_test  = np.load("y_test.npy").reshape(-1)

print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# A) Baseline (median predictor)
baseline_pred = np.full_like(y_test, fill_value=np.median(y_train), dtype=float)
print("\nBaseline (median) metrics:")
print(regression_metrics(y_test, baseline_pred))

# B) Shuffle sanity check
y_train_shuffled = y_train.copy()
np.random.shuffle(y_train_shuffled)

m = GradientBoostingRegressor(random_state=42)
m.fit(X_train, y_train_shuffled)
pred = m.predict(X_test)

print("\nShuffled-y sanity check metrics (should be BAD):")
print(regression_metrics(y_test, pred))

# C) dtype check (file size explanation)
print("\nDtypes:")
print("X_train dtype:", X_train.dtype)
print("y_train dtype:", y_train.dtype)
