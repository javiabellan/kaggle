import numpy as np
import pandas as pd
import time
import os
from functools import partial

# Feature extraction
from sklearn.pipeline        import make_pipeline
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
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
from pytorch_tabnet.tab_model import TabNetClassifier


#### TREE
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
#from sklearn.ensemble      import AdaBoostClassifier
#from sklearn.ensemble      import GradientBoostingClassifier
#from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier         # tree_method="gpu_hist" ("gpu_exact" is deprecated)
#from lightgbm              import LGBMClassifier       # 'device': 'gpu', 'gpu_platform_id': 0, and 'gpu_device_id': 0
#from catboost              import CatBoostClassifier   # task_type = 'GPU'
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
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope


########################################## CONSTANTS

results_directory = "../../results/"


print("Loading data...")
df = pd.read_csv("https://datahub.io/machine-learning/adult/r/adult.csv")
x = df.drop(["class"], axis=1)
x = df[["age", "fnlwgt", "education-num"]]
y = (df["class"]=="<=50K").astype(int)




################################################## Hyperopt library
#
#  3 algorithms are implemented:
#
#  - Random Search
#  - TPE (default)
#  - Adaptive TPE
#
#  Paramters types:
#
#  - hp.choice(label, options)
#  - hp.randint(label, upper)
#  - hp.uniform(label, low, high)
#  - hp.quniform(label, low, high, q)
#  - hp.loguniform(label, low, high)
#  - hp.qloguniform(label, low, high, q)
#  - hp.normal(label, mu, sigma)
#  - hp.qnormal(label, mu, sigma, q)
#  - hp.lognormal(label, mu, sigma)
#  - hp.qlognormal(label, mu, sigma, q)
#
# https://github.com/hyperopt/hyperopt-sklearn/blob/b9fe2d778a09f3b59c4458a058f21c970a2cec22/hpsklearn/components.py


####### Params space for each model

def uniform_int(name, min, max):
    return scope.int(hp.quniform(name, min, max, 1))

def uniform_float(name, min, max):
    return hp.uniform(name, min, max)

def log_int(name, min, max):
    assert(min)
    return scope.int(hp.qloguniform(name, np.log(min), np.log(max), 1))

def log_float(name, min, max):
    assert(min)
    return hp.loguniform(name, np.log(min), np.log(max))


params_SVM = {
    "C": hp.lognormal("C", 0, 1),
    "kernel": hp.choice("kernel", [
        {"ktype": "linear"},
        {"ktype": "RBF", "width": hp.lognormal("rbf_width", 0, 1)},
    ]),
}

params_DT = {
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "max_depth": hp.choice("max_depth",
        [None, hp.qlognormal("max_depth_int", 3, 1, 1)]),
    "min_samples_split": uniform_float("min_samples_split", 0, 1),
    #"max_features": np.random.randint(1, len(X_train.columns),20),
    #"min_samples_leaf": [2,3,4,5,6],
}

params_RF = {
    "n_estimators":  uniform_int("n_estimators", 100, 1000),
    "max_depth":     uniform_int("max_depth", 3, 15),
    "max_features":  uniform_float("max_features", 0, 1),
    "criterion":     hp.choice("criterion", ["gini", "entropy"])
}

params_XGB = {
    "max_depth":         hp.quniform("max_depth", 4, 16, 1),
    "min_child_weight":  hp.quniform("min_child", 1, 10, 1),
    "subsample":         hp.uniform("subsample", 0.7, 1),
    "gamma" :            hp.uniform("gamma", 0.1, 0.5),
    "colsample_bytree" : hp.uniform("colsample_bytree", 0.7, 1),
    "reg_lambda" :       hp.uniform("reg_lambda", 0, 1)
}




##################################### TPE (Hyperopt library)

metrics = ["Accuracy", "Acc balan", "AUC", "F1", "Kappa"]
my_results = pd.DataFrame(columns=["Model", "Hyperparameters"] + metrics + ["Time"])
print(my_results)



def train_predict(model, x, y, validation):

    if validation=="cv":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        oof_pred_proba = cross_val_predict(model, x, y, cv=skf, method="predict_proba")
        
        real       = y
        pred_proba = oof_pred_proba[:,1]
        pred       = np.argmax(oof_pred_proba, axis=1)
    
    if validation=="split":
        x_train, x_val, y_train, y_val = train_test_split(x, y,
                                            test_size=0.33,
                                            stratify=y,
                                            shuffle=True,
                                            random_state=0)

        model.fit(x_train, y_train)
        oof_pred_proba = model.predict_proba(x_val)

        real       = y_val
        pred_proba = oof_pred_proba[:,1]
        pred       = np.argmax(oof_pred_proba, axis=1)

    return real, pred, pred_proba



def func2minimize(params, model_name, model_fn): 
    
    model = model_fn(**params)

    start_time = time.time()
    real, pred, pred_proba = train_predict(model, x, y, validation="split")
    total_time = time.time() - start_time

    global my_results    
    my_results = my_results.append({
                    "Model":      model_name,
                    "Hyperparameters": params,
                    "Accuracy":   accuracy_score(real, pred),
                    "Acc balan":  balanced_accuracy_score(real, pred),
                    "AUC":        roc_auc_score(real, pred_proba),
                    "F1":         f1_score(real, pred),
                    #"Recall":    recall_score(real, pred),
                    #"Precision": precision_score(real, pred),
                    "Kappa":      cohen_kappa_score(real, pred),
                    "Time":       total_time},
                    ignore_index=True)

    return -balanced_accuracy_score(real, pred)



############################################################ MODELS

models = [
    ("DT",   DecisionTreeClassifier,         params_DT,   20),
    ("RF",   RandomForestClassifier,         params_RF,   1),
#   ("ET",   ExtraTreesClassifier,           params_ET,   0),
#   ("AB",   AdaBoostClassifier,             params_AB,   0),
#   ("GB",   GradientBoostingClassifier,     params_GB,   0),
#   ("HGB",  HistGradientBoostingClassifier, params_HGB,  0),
#   ("XGB",  XGBClassifier,                  params_XGB,  0),
#   ("LGBM", LGBMClassifier,                 params_LGBM, 0),
#   ("CB",   CatBoostClassifier,             params_CB,   0),
#   ("NGB",  NGBClassifier,                  params_NGB,  0),
#   ("RGF",  RGFClassifier,                  params_RGF,  0),
#   ("FRGF", FastRGFClassifier,              params_FRGF, 0)
]


for model_name, model_fn, model_params, model_evals in models:

    print(f"{model_name}...")
    optimizeModel = partial(func2minimize, model_name=model_name, model_fn=model_fn)

    result = fmin(fn=optimizeModel,
                  space=model_params,
                  algo=tpe.suggest,
                  max_evals=model_evals,
                  trials=Trials())

    print(result) # -> {"a": 1, "c2": 0.01420615366247227}


#import hyperopt
#print(hyperopt.space_eval(params_space, result))  # -> ("case 2", 0.01420615366247227}


################################ Save HTML

def highlight_max(s):
    is_max = s == s.max()
    return ["font-weight: bold" if v else "" for v in is_max]
     
html = my_results.style.set_precision(8)\
               .bar(subset=metrics, vmin=0, vmax=1, color="rgba(95, 186, 125, 0.8)")\
               .apply(highlight_max, subset=metrics)\
               .set_caption("Models Metrics.")\
               .render()

# SORTABLE
# https://www.kryogenix.org/code/browser/sorttable/
sort_script = "<script src=\"https://www.kryogenix.org/code/browser/sorttable/sorttable.js\"></script>"
html = sort_script + html
html = html.replace("<table", "<table class=\"sortable\"")

# Save
tfile = open("../../results/hyperopt_results.html", "w")
tfile.write(html)
tfile.close()