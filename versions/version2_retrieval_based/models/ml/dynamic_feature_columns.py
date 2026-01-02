import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from data_module import HelocDataModule
from models.orbit.config import OrbitConfig
from evaluators.metric import calculate_classification_metrics

# Load the dataset
config = OrbitConfig()
data_module = HelocDataModule(config)

X_train, y_train = data_module.get_feature_dataset(data_module.train_indices)
X_test, y_test = data_module.get_feature_dataset(data_module.val_indices)

# Models
models = {
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "NaiveBayes": GaussianNB(),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, verbosity=1),
}


# Preprocessor: Scale only the first 23 features, leave the rest unchanged
transformers = []
if X_train.shape[1] <= 23:
    transformers.append(("scale", StandardScaler(), slice(0, X_train.shape[1])))
else:
    transformers.append(("scale", StandardScaler(), slice(0, 23)))
    transformers.append(("skip", "passthrough", slice(23, X_train.shape[1])))

preprocessor = ColumnTransformer(
    transformers=transformers
)

results = []
results_drop = []

for name, model in models.items():
    print("=" * 40)
    print(f"Training {name} ...")
    
    # For tree-based models, scaling is not strictly necessary 
    # But using Pipeline for consistency and easier management
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Evaluation
    result = calculate_classification_metrics(y_test, y_pred, y_proba)
    results.append({
        "Model": name,
        **result,
    })
    
    # Evaluation with missing value in features
    X_test_drop = X_test.copy()
    X_test_drop[:, 8:12] = -7 # 8:12
    y_pred_drop = clf.predict(X_test_drop)
    y_proba_drop = clf.predict_proba(X_test_drop)[:, 1]
    result_drop = calculate_classification_metrics(y_test, y_pred_drop, y_proba_drop)
    results_drop.append({
        "Model": name,
        **result_drop,
    })

# Display results
results_df = pd.DataFrame(results)
print("\nSummary of model performance:")
print(results_df)

# Save results to CSV
results_df.to_csv("model_results.csv", index=False)
print("\n✅ Results have been saved to model_results.csv")

results_df_drop = pd.DataFrame(results_drop)
print("\nSummary of model performance with missing value in feature 5:")
print(results_df_drop)

# Save results to CSV
results_df_drop.to_csv("model_results_drop_feature5.csv", index=False)
print("\n✅ Results with missing value have been saved to model_results_drop_feature5.csv")
