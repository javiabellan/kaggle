import pandas as pd
import numpy as np
from scipy.stats import gmean
from scipy.optimize import fmin
from sklearn.metrics import roc_auc_score



####### Get data (model oof predictions)

df = pd.read_csv("oof_preds.csv")

y = df["target"]
model_preds = df.drop(columns=["target"])
n_models = len(model_preds.columns)



############################################ Ensamble methods
ensemble_preds   = pd.DataFrame({})


ensemble_preds["mean"]           = np.mean(model_preds.values, axis=1)
ensemble_preds["geometric_mean"] = gmean(model_preds.values, axis=1)
ensemble_preds["median"]         = np.median(model_preds.values, axis=1)
ensemble_preds["rank_mean"]      = model_preds.rank().values.mean(axis=1)

#### Weighted average
def func2minimize(weights):
	pred = np.sum(model_preds * weights, axis=1)
	auc  = roc_auc_score(y, pred)
	return -auc

init_weights = np.ones(n_models) / n_models
weights      = fmin(func2minimize, init_weights)
ensemble_preds["weighted_mean"] = np.sum(model_preds * weights, axis=1)



############################################ Ensamble result metrics

all_preds = pd.concat([model_preds, ensemble_preds], axis=1, sort=False)
ensemble_results = pd.DataFrame({'Method': [], "AUC": []})


for model_name in all_preds.columns:
    ensemble_results = ensemble_results.append(
                        {"Method": model_name,
                         "AUC":    roc_auc_score(y, all_preds[model_name])},
                         ignore_index=True)
        
ensemble_results = ensemble_results.sort_values(by=["AUC"], ascending=False, ignore_index=True)

print(ensemble_results)

ensemble_preds.to_csv("ensemble_preds.csv", index=False)
ensemble_results.to_csv("ensemble_results.csv", index=False)