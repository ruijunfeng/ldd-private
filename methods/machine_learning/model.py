import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils.json_utils import save_json

from data_module import HelocDataModule

# Load the dataset
data_module = HelocDataModule()
X_train, y_train = data_module.get_feature_dataset(data_module.train_indices)
X_val, y_val = data_module.get_feature_dataset(data_module.val_indices)
X_test, y_test = data_module.get_feature_dataset(data_module.test_indices)

# Models
models = {
    # Basic Models
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    # Ensemble Methods (Bagging and Boosting)
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, verbosity=-1),
    "LightGBM": LGBMClassifier(random_state=42, verbosity=-1),
}

# Train and evaluate each model
for name, model in models.items():
    print("=" * 40)
    print(f"Training {name} ...")
    
    # For tree-based models, scaling is not strictly necessary 
    # But using Pipeline for consistency and easier management
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("model", model)
    ])
    clf.fit(X_train, y_train)
    
    # Results
    model_results = {}
    
    # Validation
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]
    for i, idx in enumerate(data_module.val_indices):
        model_results[str(idx)] = {
            "y_pred": int(y_pred[i]),
            "y_proba": float(y_proba[i])
        }
    
    # Test
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    for i, idx in enumerate(data_module.test_indices):
        model_results[str(idx)] = {
            "y_pred": int(y_pred[i]),
            "y_proba": float(y_proba[i])
        }
    
    # Create the output directory
    output_dir = f"results/machine_learning/{name}"
    os.makedirs(output_dir, exist_ok=True)
    save_json(f"{output_dir}/predictions.json", model_results)
