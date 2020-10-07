import time
import pandas as pd
import numpy as np
from scipy.stats import gmean, hmean
from scipy.optimize import fmin

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Metamodels
from sklearn.linear_model  import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.base import BaseEstimator


####### Get data (model oof predictions)

df = pd.read_csv("../../results/model_preds.csv")

y = df["target"]
model_preds = df.drop(columns=["target"])
x = model_preds.values
n_models = len(model_preds.columns)



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

ensemble_preds_prob = pd.DataFrame({})

############################################ Simle ensamble methods

ensemble_preds_prob["Arithmetic mean"] = np.mean(model_preds.values, axis=1)
ensemble_preds_prob["Geometric mean"]  = gmean(model_preds.values, axis=1)
ensemble_preds_prob["Harmonic mean"]   = hmean(model_preds.values, axis=1)
ensemble_preds_prob["Median"]          = np.median(model_preds.values, axis=1)
#ensemble_preds_prob["Rank mean"]       = model_preds.rank().values.mean(axis=1)


############################################ STACKING ensamble methods

class WeightedMean(BaseEstimator):
	def __init__(self):
		self.weights = -1

	def fit(self, x, y):
		def func2minimize(w):
			pred = np.sum(x * w, axis=1)
			auc  = roc_auc_score(y, pred)
			return -auc
		n_models = x.shape[1]
		init_weights = np.ones(n_models) / n_models
		self.weights = fmin(func2minimize, init_weights, disp=False)

	def predict(self, x):
		return np.sum(x * self.weights, axis=1)

stacking_metamodels = {
  "STACKING Weighted mean":  WeightedMean(),
  "STACKING LinRegr":        LinearRegression(),
  "STACKING LinRegr noBias": LinearRegression(fit_intercept=False),
  "STACKING Ridge":          Ridge(),
  "STACKING Ridge noBias":   Ridge(fit_intercept=False),
  #"STACKING Lasso":          Lasso(),
  #"STACKING Lasso noBias":   Lasso(fit_intercept=False),
  #"STACKING ElasticNet":     ElasticNet(),
  #"STACKING ElasticNet":     ElasticNet(fit_intercept=False)
}

for model_name, model in stacking_metamodels.items():

	start_time = time.time()
	oof_pred_prob = cross_val_predict(model, x, y, cv=skf)
	#oof_pred_prob = cross_val_predict(model, x, y, cv=skf, method="predict_proba")[:,1]
	total_time = time.time() - start_time

	ensemble_preds_prob[model_name] = oof_pred_prob


ensemble_preds_prob.to_csv("../../results/ensemble_oof_preds.csv", index=False)


############################################ Ensamble result metrics

all_preds_prob = pd.concat([model_preds, ensemble_preds_prob], axis=1, sort=False)
all_preds_abs  = (all_preds_prob>0.5).astype(int)

metrics = ["Accuracy", "Acc balan", "AUC", "F1", "Recall", "Precision", "Kappa"]
results = pd.DataFrame(columns=["Model"] + metrics + ["Time"])


for model_name in all_preds_prob.columns:

	pred_proba = all_preds_prob[model_name]
	pred_abs   = all_preds_abs[model_name]

	results = results.append({"Model":     model_name,
                              "Accuracy":  accuracy_score(y, pred_abs),
                              "Acc balan": balanced_accuracy_score(y, pred_abs),
                              "AUC":       roc_auc_score(y, pred_proba),
                              "F1":        f1_score(y, pred_abs),
                              "Recall":    recall_score(y, pred_abs),
                              "Precision": precision_score(y, pred_abs),
                              "Kappa":     cohen_kappa_score(y, pred_abs),
                              "Time":      total_time},
                              ignore_index=True)
print(results)


###################################### plot results

table_props = [
  #("margin-left", "auto"),
  #("margin-right", "auto"),
  ("border", "none"),
  ("border-collapse", "collapse"),
  ("border-spacing", "0"),
  #("font-size", "12px"),
  #("table-layout", "fixed")
  #("table-layout", "fixed"),
  #("overflow-x", "scroll"),
  ("white-space", "nowrap")
  #margin:40px auto 0px auto;
]

thead_props = [
  ("background-color", "#eee"),
  ("color", "#666666"),
  ("font-weight", "bold"),
  ("cursor", "pointer"),
]

th_props = [
  ('font-size', '16px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', '#6d6d6d'),
  ('background-color', '#f7f7f9')
]

td_props = [
  ("font-family", "Arial"),
  ('font-size', '14px'),
  ("onclick", "sortTable(0)")
]



css_styles = [
  dict(selector="table", props=table_props),
  dict(selector="thead", props=thead_props),
 # dict(selector="th", props=th_props),
 # dict(selector="td", props=td_props),
 # dict(selector="tr:hover", props=[("background-color", "#eeeeee")]),
 # dict(selector="caption",  props=[("caption-side", "bottom")])
]


css_styles = [
    #table properties
    dict(selector=" ", 
         props=[("margin","0"),
                ("font-family",'"Helvetica", "Arial", sans-serif'),
                ("border-collapse", "collapse"),
                ("border","none"),
#                 ("border", "2px solid #ccf")
                   ]),

    # header color - optional
    dict(selector="thead", 
         props=[("background-color","#eee"),
                ("color", "#666666"),
                ("cursor", "pointer")]),
    
    # background shading
   # dict(selector="tbody tr:nth-child(odd)",
   #      props=[("background-color", "#fff")]),
   # dict(selector="tbody tr:nth-child(even)",
   #      props=[("background-color", "#eee")]),

    # Hover selection
    dict(selector="tr:hover", props=[("background-color", "#ddd")]),

    # header cell properties
    dict(selector="th", 
         props=[("font-size", "100%"),
                ("text-align", "center"),
                ("padding", ".25em")]),

    # cell spacing
    dict(selector="td", 
         props=[("padding", ".5em")]),
]

def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]
     
#              .bar(subset=metrics, vmin=0, vmax=1, color='#5fba7d')\  

html = results.style.set_precision(8)\
               .bar(subset=metrics, vmin=0, vmax=1, color='rgba(95, 186, 125, 0.8)')\
               .apply(highlight_max, subset=metrics)\
               .set_caption('Models Metrics.')\
               .set_table_styles(css_styles)\
               .render()

# SORTABLE
# https://www.kryogenix.org/code/browser/sorttable/
sort_script = "<script src=\"https://www.kryogenix.org/code/browser/sorttable/sorttable.js\"></script>"
html = sort_script + html
html = html.replace("<table", "<table class=\"sortable\"")

# Save
tfile = open("../../results/ensemble_oof_results.html", 'w')
tfile.write(html)
tfile.close()


