import os
import torch
import pandas as pd

from utils.json_utils import load_json
from evaluators.metric import calculate_classification_metrics

from models.orbit.config import OrbitConfig
from data_module import HelocDataModule

# prepare the results for analysis
config = OrbitConfig()
data_module = HelocDataModule(config)
_, labels = data_module.get_profile_dataset(data_module.val_indices)
y_true = torch.tensor(labels)

# Save the results of different ablation settings
results = []

# Orbit
orbit_results = load_json("results/orbit/version_53/predictions.json")
y_pred = torch.tensor([orbit_results[str(i)]["y_pred"][0] for i in data_module.val_indices])
y_proba = torch.tensor([orbit_results[str(i)]["y_proba"][0] for i in data_module.val_indices])
result = calculate_classification_metrics(y_true, y_pred, y_proba)
results.append(("Orbit", result))

# convert into dataframe
df = pd.DataFrame([r for _, r in results], index=[name for name, _ in results])

# save as csv
output_dir = "results/summary"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f"{output_dir}/performance_analysis.csv")
