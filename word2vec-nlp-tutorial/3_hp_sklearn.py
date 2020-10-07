import numpy as np
import pandas as pd
import time
import os

# Feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline        import make_pipeline
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.experimental    import enable_hist_gradient_boosting

########################################################### CLASSIFIERS

#### MULT
from sklearn.linear_model   import LogisticRegression
from sklearn.linear_model   import RidgeClassifier
from sklearn.svm            import SVC
from sklearn.svm            import NuSVC
from sklearn.svm            import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes    import GaussianNB
from sklearn.naive_bayes    import MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble       import StackingClassifier

#### TREE
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
#from xgboost               import XGBClassifier, plot_tree
#from lightgbm              import LGBMClassifier
#from catboost              import CatBoostClassifier
#from ngboost               import NGBClassifier
#from rgf.sklearn           import RGFClassifier, FastRGFClassifier


########################################################### METRICS
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import cohen_kappa_score

# Hyperparameters optimization
from sklearn.model_selection import GridSearchCV       # 1) Grid Search
from sklearn.model_selection import RandomizedSearchCV # 2) Random Search


########################################## CONSTANTS

results_directory = "../../results/"





########################################## Read the dataset

df = pd.read_csv("../../data/labeledTrainData.tsv", sep='\t')[:100]
x = df["review"].astype(str)
y = df["sentiment"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


########################################## Define model

model = Pipeline([('tfidf', TfidfVectorizer()),
                  ('rf', RandomForestClassifier(n_jobs=-1))])



########################################## 1) Grid Search (Sklearn library)
params = {
    "rf__n_estimators": [100, 200], # 300
    "rf__max_depth":    [1, 3, 5], # 7
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(estimator=model,
                           param_grid=params,
                           scoring="accuracy",
                           n_jobs=-1,
                           cv=skf,
                           verbose=10)

grid_search.fit(x, y)

print(grid_search.best_score_) # Returns (float) Mean cross-validated score of the best_estimator
print(grid_search.best_params_) # (dict) Best Parameter setting
print(grid_search.best_estimator_) # returns estimator

########################################## 2) Random Search (Sklearn library)

params = {
    "rf__n_estimators": np.arange(100, 1500, 100), # 300
    "rf__max_depth":    np.arange(1, 20, 1)
    "criterion": ["gini", "entropy"]
}

random_search = RandomizedSearchCV(estimator=model,
                           param_distributions=params,
                           n_iter=10,
                           scoring="accuracy",
                           n_jobs=-1,
                           cv=skf,
                           verbose=10)

random_search.fit(x, y)

print(random_search.best_score_) # Returns (float) Mean cross-validated score of the best_estimator
print(random_search.best_params_) # (dict) Best Parameter setting
print(random_search.best_estimator_) # returns estimator





