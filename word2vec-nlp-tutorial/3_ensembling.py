import pandas as pd
import numpy as np
from scipy.stats import gmean, hmean
from scipy.optimize import fmin
from sklearn.metrics import roc_auc_score


# Metamodels
from sklearn.linear_model  import LinearRegression, Ridge, ElasticNet


####### Get data (model oof predictions)

df = pd.read_csv("../../results/model_preds.csv")

y = df["target"]
model_preds = df.drop(columns=["target"])
x = model_preds.values
n_models = len(model_preds.columns)



############################################ Ensamble methods
ensemble_preds   = pd.DataFrame({})


ensemble_preds["Arithmetic mean"] = np.mean(model_preds.values, axis=1)
ensemble_preds["Geometric mean"]  = gmean(model_preds.values, axis=1)
ensemble_preds["Harmonic mean"]   = hmean(model_preds.values, axis=1)
ensemble_preds["Median"]          = np.median(model_preds.values, axis=1)
ensemble_preds["Rank mean"]       = model_preds.rank().values.mean(axis=1)

#### Weighted average
def func2minimize(weights):
	pred = np.sum(model_preds * weights, axis=1)
	auc  = roc_auc_score(y, pred)
	return -auc

init_weights = np.ones(n_models) / n_models
weights      = fmin(func2minimize, init_weights, disp=False)
ensemble_preds["Weighted mean"] = np.sum(model_preds * weights, axis=1)

print("Weighted mean weights=", weights)




############################################ STACKING

linRegr = LinearRegression()
linRegr.fit(x, y)
ensemble_preds["STACKING LinRegr"] = linRegr.predict(x)
print("STACKING LinRegr weights=", linRegr.coef_, "B=", linRegr.intercept_)

ridge = Ridge()
ridge.fit(x, y)
ensemble_preds["STACKING Ridge"] = ridge.predict(x)
print("STACKING Ridge weights=", ridge.coef_, "B=", ridge.intercept_)

elasticNet = ElasticNet()
elasticNet.fit(x, y)
ensemble_preds["STACKING ElasticNet"] = elasticNet.predict(x)
print("STACKING ElasticNet weights=", elasticNet.coef_, "B=", elasticNet.intercept_)



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

ensemble_preds.to_csv("../../results/ensemble_preds.csv", index=False)
ensemble_results.to_csv("../../results/ensemble_results.csv", index=False)