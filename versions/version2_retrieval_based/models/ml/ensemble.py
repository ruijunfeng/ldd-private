import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from data_module import HelocDataModule
from models.orbit.config import OrbitConfig
from evaluators.metric import calculate_classification_metrics

# Load the dataset
config = OrbitConfig()
data_module = HelocDataModule(config)

X_train, y_train = data_module.get_feature_dataset(data_module.train_indices)
X_test, y_test = data_module.get_feature_dataset(data_module.val_indices)

# Classifier
model = GradientBoostingClassifier(random_state=42)
results = []

# Using Pipeline for consistency
clf = Pipeline([
    ("scale", StandardScaler()),
    ("model", model)
])
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Evaluation
result = calculate_classification_metrics(y_test, y_pred, y_proba)
results.append({
    "Model": "GBDT",
    **result,
})


# Sementic Stacking
import torch
import numpy as np

# Initialize parameters
top_k = 5
ranked_indices = data_module.train_ranked_indices
similarity_scores = data_module.train_similarity_scores
topk_similarity_scores = torch.gather(similarity_scores, 1, ranked_indices[:, :top_k]).numpy()

# Retrieve predicted probabilities of similar instances from training set
y_proba_train = clf.predict_proba(X_train)[:, 1]

topk_y_proba = []
for ranked_index in ranked_indices:
    topk_y_proba.append(y_proba_train[ranked_index[:top_k].numpy()])

topk_y_proba = np.array(topk_y_proba)
topk_proba = (topk_y_proba * topk_similarity_scores).mean(axis=1)

# Train a logistic regression model on the top-k probabilities
X_meta_train = np.hstack([y_proba_train.reshape(-1, 1), topk_proba.reshape(-1, 1)])
meta_model = LogisticRegression(random_state=42)
meta_model.fit(X_meta_train, y_train)


# Retrieve predicted probabilities of similar instances from training set (validation set)
ranked_indices = data_module.val_ranked_indices
similarity_scores = data_module.val_similarity_scores
topk_similarity_scores = torch.gather(similarity_scores, 1, ranked_indices[:, :top_k]).numpy()

topk_y_proba = []
for ranked_index in ranked_indices:
    topk_y_proba.append(y_proba_train[ranked_index[:top_k].numpy()])

topk_y_proba = np.array(topk_y_proba)
topk_proba = (topk_y_proba * topk_similarity_scores).mean(axis=1)

# Evaluate the meta-model
y_proba_val = clf.predict_proba(X_test)[:, 1]
X_meta_val = np.hstack([y_proba_val.reshape(-1, 1), topk_proba.reshape(-1, 1)])
y_ens_pred = meta_model.predict(X_meta_val)
y_ens_proba = meta_model.predict_proba(X_meta_val)[:, 1]
result = calculate_classification_metrics(y_test, y_ens_pred, y_ens_proba)
results.append({
    "Model": "GBDT_Sementic_Stacking",
    **result,
})


# Sementic Stacking (direct weighted average of predicted probabilities from similar instances)
import torch
import numpy as np

# Initialize parameters
top_k = 200
ranked_indices = data_module.val_ranked_indices
similarity_scores = data_module.val_similarity_scores
topk_similarity_scores = torch.gather(similarity_scores, 1, ranked_indices[:, :top_k]).numpy()

# Retrieve predicted probabilities of similar instances from training set
y_proba_train = clf.predict_proba(X_train)[:, 1]

topk_y_proba = []
for ranked_index in ranked_indices:
    topk_y_proba.append(y_proba_train[ranked_index[:top_k].numpy()])

topk_y_proba = np.array(topk_y_proba)

# Compute final ensemble probabilities
topk_proba = (topk_y_proba * topk_similarity_scores).mean(axis=1)
y_proba = clf.predict_proba(X_test)[:, 1]

y_ens_proba = 0.5 * y_proba + 0.5 * topk_proba
y_ens_pred = (y_ens_proba >= 0.5).astype(int)

result = calculate_classification_metrics(y_test, y_ens_pred, y_ens_proba)
results.append({
    "Model": "GBDT_Sementic_Average",
    **result,
})


# Display results
results_df = pd.DataFrame(results)
print("\nSummary of model performance:")
print(results_df)

# Save results to CSV
results_df.to_csv("model_results_gbdt.csv", index=False)
print("\n✅ Results have been saved to model_results_gbdt.csv")