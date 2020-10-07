import numpy as np
import pandas as pd
import time
import os

# Feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.naive_bayes   import MultinomialNB
from sklearn.pipeline      import make_pipeline

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import StratifiedKFold, cross_val_predict



df = pd.read_csv("../../data/labeledTrainData.tsv", sep='\t')
x = df["review"].astype(str)
y = df["sentiment"]


models = {
  #"lr_bow":   make_pipeline(CountVectorizer(), LogisticRegression()),
  "lr_tfidf": make_pipeline(TfidfVectorizer(), LogisticRegression()),
  "nb_bow":   make_pipeline(CountVectorizer(), MultinomialNB()),
  "nb_tfidf": make_pipeline(TfidfVectorizer(), MultinomialNB()),
  #"rf_bow":   make_pipeline(CountVectorizer(), RandomForestClassifier()),
  #"rf_tfidf": make_pipeline(TfidfVectorizer(), RandomForestClassifier()),
}

"""
# Multiclass classification
accuracy  = metrics.accuracy_score(ytest,pred_)
acc_balan = metrics.balanced_accuracy_score(ytest,pred_)
recall    = metrics.recall_score(ytest,pred_, average='macro')
precision = metrics.precision_score(ytest,pred_, average = 'weighted')
f1_weight = metrics.f1_score(ytest,pred_, average='weighted')
f1_macro  = metrics.f1_score(ytest,pred_, average='macro')
kappa     = metrics.cohen_kappa_score(ytest,pred_)
mcc       = metrics.matthews_corrcoef(ytest,pred_)

## Binary classification
accuracy  = metrics.accuracy_score(ytest,pred_)
auc       = metrics.roc_auc_score(ytest,pred_prob)
recall    = metrics.recall_score(ytest,pred_)
precision = metrics.precision_score(ytest,pred_)
f1        = metrics.f1_score(ytest,pred_)
kappa     = metrics.cohen_kappa_score(ytest,pred_)
mcc       = metrics.matthews_corrcoef(ytest,pred_)
"""

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
metrics = ["Accuracy", "Acc balan", "AUC", "F1", "Recall", "Precision", "Kappa"]

preds   = pd.DataFrame({"target": y})
results = pd.DataFrame(columns=["Model"] + metrics + ["Time"])


for name, model in models.items():

	print(f"{name}...")   

	start_time = time.time()
	pred_proba = cross_val_predict(model, x, y, cv=skf, method="predict_proba") # Only for binarr clasification
	pred       = np.argmax(pred_proba, axis=1)
	total_time = time.time() - start_time

	preds[name] = pred_proba[:,1]

	results = results.append({"Model":     name,
                              "Accuracy":  accuracy_score(y, pred),
                              "Acc balan": balanced_accuracy_score(y, pred),
                              "AUC":       roc_auc_score(y, pred_proba[:,1]),
                              "F1":        f1_score(y, pred),
                              "Recall":    recall_score(y, pred),
                              "Precision": precision_score(y, pred),
                              "Kappa":     cohen_kappa_score(y, pred),
                              "Time":      total_time},
                              ignore_index=True)

	os.system('clear')
	print(results)
	print("")


results = results.sort_values(by=["AUC"], ascending=False, ignore_index=True)

preds.to_csv("../../results/model_preds.csv", index=False)
results.to_csv("../../results/model_results.csv", index=False)



###################################### plot results

thead_props = [
  ("background-color", "#eee"),
  ("color", "#666666"),
  ("font-weight", "bold"),
  ("cursor", "pointer"),
]

css_styles = [
  dict(selector="thead", props=thead_props),
]

def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

html = results.style.set_precision(8)\
              .bar(subset=metrics, vmin=0, vmax=1, color='#5fba7d')\
              .apply(highlight_max, subset=metrics)\
              .set_caption('Models Metrics.')\
              .set_table_styles(css_styles)\
              .render()

# Make sortable (https://www.kryogenix.org/code/browser/sorttable)
sort_script = "<script src=\"https://www.kryogenix.org/code/browser/sorttable/sorttable.js\"></script>"
html = sort_script + html
html = html.replace("<table", "<table class=\"sortable\"")

# Save
file = open("../../results/model_results.html", 'w')
file.write(html)
file.close()