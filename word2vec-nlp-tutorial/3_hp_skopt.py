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
from skopt import gp_minimize                          # 3) Bayesian with GP (lib: skopt)
from skopt import space 
from skopt import utils
from hyperopt import hp, fmin, tpe, Trials             # 4) TPE (lib: hyperopt)
from hyperopt.pyll.base import scope
import optuna


########################################## CONSTANTS

results_directory = "../../results/"





########################################## Read the dataset

df = pd.read_csv("../../data/labeledTrainData.tsv", sep='\t')[:100]
x = df["review"].astype(str)
y = df["sentiment"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


 
################################# 3) Bayesian opt with GP (skopt library)

# https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html

# Params space
# https://scikit-optimize.github.io/stable/modules/classes.html#module-skopt.space.space


params_space = [
    space.Integer(3, 15, name="max_depth"),
    space.Integer(100, 1000, name="n_estimators"),
    space.Real(0.01, 1, prior="uniform", name="max_features"),
    space.Categorical(["gini", "entropy"], name="criterion")
]


@utils.use_named_args(params_space)
def func2minimize(**params): 
    #params = dict(zip(param_names, params))
    model = Pipeline([('tfidf', TfidfVectorizer()),
                      ('rf', RandomForestClassifier(**params))])

    oof_pred = cross_val_predict(model, x, y, cv=skf)
    acc = accuracy_score(y, oof_pred)
    return -acc


result = gp_minimize(func=func2minimize,
                     dimensions=params_space,
                     n_calls=15,
                     n_initial_points=10,
                     verbose=True)

print(result.x)
