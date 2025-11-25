import os
import numpy as np
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
FEATURES_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/features"))

X_path = os.path.join(FEATURES_DIR, "X_features.npy")
y_path = os.path.join(FEATURES_DIR, "y_labels.npy")

print("Loading:", X_path)
print("Loading:", y_path)


X = np.load(X_path)
y = np.load(y_path)


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp)


np.save(os.path.join(FEATURES_DIR, "X_train.npy"), X_train)
np.save(os.path.join(FEATURES_DIR, "y_train.npy"), y_train)
np.save(os.path.join(FEATURES_DIR, "X_val.npy"), X_val)
np.save(os.path.join(FEATURES_DIR, "y_val.npy"), y_val)
np.save(os.path.join(FEATURES_DIR, "X_test.npy"), X_test)
np.save(os.path.join(FEATURES_DIR, "y_test.npy"), y_test)

print("Split complete!")
