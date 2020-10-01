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
from sklearn.metrics       import accuracy_score
from sklearn.metrics       import balanced_accuracy_score
from sklearn.metrics       import roc_auc_score


from sklearn.model_selection import StratifiedKFold, cross_val_predict


df = pd.read_csv("../data/labeledTrainData.tsv", sep='\t')
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



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

preds   = pd.DataFrame({"target": y})
results = pd.DataFrame({'Model': [], "AUC": [], 'Time': []})

for name, model in models.items():

	print(f"{name}...")   

	start_time = time.time()
	#pred = cross_val_predict(model, x, y, cv=skf)
	pred = cross_val_predict(model, x, y, cv=skf, method="predict_proba")[:,1] # Only for binarr clasification
	total_time = time.time() - start_time
	preds[name] = pred

	results = results.append({"Model":    name,
                              "AUC":      roc_auc_score(y, pred),
                              "Time":     total_time},
                              ignore_index=True)

	os.system('clear')
	print(results)
	print("")


preds.to_csv("oof_preds.csv", index=False)
results.to_csv("results.csv", index=False)