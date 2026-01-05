import os
import torch
import pandas as pd

from utils.json_utils import load_json
from evaluators.metric import calculate_classification_metrics

from data_module import HelocDataModule

# prepare the results for analysis
data_module = HelocDataModule()
y_true = torch.tensor([item["labels"] for item in data_module.get_profile_dataset(data_module.test_indices)])

# save the results of different methods
results = []

# Machine Learning Models
model_names = [
    "KNN", "MLP", "LogisticRegression", "SVM", "NaiveBayes", "DecisionTree", 
    "RandomForest", "GradientBoosting", "XGBoost", "LightGBM",
]
for name in model_names:
    model_results = load_json(f"results/machine_learning/{name}/predictions.json")
    y_pred = torch.tensor([model_results[str(i)]["y_pred"] for i in data_module.test_indices])
    y_proba = torch.tensor([model_results[str(i)]["y_proba"] for i in data_module.test_indices])
    result = calculate_classification_metrics(y_true, y_pred, y_proba)
    results.append((name, result))

# TabLLM (Zero-Shot Prompting)
tabllm_results = load_json("results/tabllm/predictions.json")
y_pred = torch.tensor([tabllm_results[str(i)]["y_pred"] for i in data_module.test_indices])
y_proba = torch.tensor([tabllm_results[str(i)]["y_proba"] for i in data_module.test_indices])
result = calculate_classification_metrics(y_true, y_pred, y_proba)
results.append(("TabLLM", result))

# CALM (LoRA)
lora_results = load_json("results/calm/version_0/predictions.json")
y_pred = torch.tensor([int(lora_results[str(i)]["y_proba"] >=0.5) for i in data_module.test_indices])
y_proba = torch.tensor([lora_results[str(i)]["y_proba"] for i in data_module.test_indices])
result = calculate_classification_metrics(y_true, y_pred, y_proba)
results.append(("CALM", result))

# SNAP
snap_results = load_json("results/snap/full_model/version_0/predictions.json")
y_pred = torch.tensor([int(snap_results[str(i)]["y_proba"] >=0.5) for i in data_module.test_indices])
y_proba = torch.tensor([snap_results[str(i)]["y_proba"] for i in data_module.test_indices])
result = calculate_classification_metrics(y_true, y_pred, y_proba)
results.append(("SNAP", result))

# convert into dataframe
df = pd.DataFrame([r for _, r in results], index=[name for name, _ in results])

# save as csv
output_dir = "results/summary"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f"{output_dir}/performance_analysis.csv")
