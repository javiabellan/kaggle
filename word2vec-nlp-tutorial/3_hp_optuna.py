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


################################################## Optuna library
#
# Algorithms
#
# - optuna.samplers.GridSampler    Sampler using grid search.
# - optuna.samplers.RandomSampler  Sampler using random sampling.
# - optuna.samplers.TPESampler     Sampler using TPE algorithm. (defalt)
# - optuna.samplers.CmaEsSampler   Sampler using CMA-ES algorithm.
#
#
# Paramters types:
#
# - suggest_categorical(name, choices)              categorical parameter.
# - suggest_float(name, low, high, *[, step, log])  floating point parameter.
# - suggest_int(name, low, high[, step, log])       integer parameter.
# - suggest_uniform(name, low, high)                continuous parameter.
# - suggest_discrete_uniform(name, low, high, q)    discrete parameter.
# - suggest_loguniform(name, low, high)             continuous parameter.
#
# should_prune():                                 Suggest whether the trial should be pruned or not.


metrics = ["Accuracy", "Acc balan", "F1", "Kappa"]
my_results = pd.DataFrame(columns=["Model", "Parameters"] + metrics + ["Time"])
print(my_results)

def objective(trial):

    model_name = trial.suggest_categorical("model", ["LR", "SVC", "RF", "KNN"])
    print(f"{model_name}...")   


    if model_name == 'LR':
        model = LogisticRegression(
            C            = trial.suggest_float("logreg_c", 1e-6, 10, log=True),
            penalty      = trial.suggest_categorical("logreg_penalty", ["l1", "l2"]),
            solver       = 'liblinear',
            class_weight = trial.suggest_categorical("logreg_class_weight", ["balanced", None]),
        )

    if model_name == "SVC":
        model = SVC(
            C      = trial.suggest_float("svc_c", 1e-6, 10, log=True),
            gamma  = "auto"
        )

    if model_name == "KNN":
        model = KNeighborsClassifier(
            n_neighbors = trial.suggest_int("knn_neighbors", 1, 50),
            weights     = trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
            metric      = trial.suggest_categorical("knn_metric", ["euclidean", "manhattan"])
        )

    if model_name == "RF":
        model = RandomForestClassifier(
            n_estimators = trial.suggest_int("rf_n_estimators", 100, 1000),
            max_depth    = trial.suggest_int("rf_max_depth", 2, 32, log=True),
            max_features = trial.suggest_uniform("max_features", 0.01, 1),
            criterion    = trial.suggest_categorical("criterion", ["gini", "entropy"])
        )

    model = make_pipeline(TfidfVectorizer(), model)

    start_time = time.time()
    oof_pred = cross_val_predict(model, x, y, cv=skf)
    total_time = time.time() - start_time

    global my_results
    print(my_results)
    
    my_results = my_results.append({"Model":     model_name,
                              "Parameters":"TO-DO",
                              "Accuracy":  accuracy_score(y, oof_pred),
                              "Acc balan": balanced_accuracy_score(y, oof_pred),
                              #"AUC":       roc_auc_score(y, pred_proba[:,1]),
                              "F1":        f1_score(y, oof_pred),
                              #"Recall":    recall_score(y, oof_pred),
                              #"Precision": precision_score(y, oof_pred),
                              "Kappa":     cohen_kappa_score(y, oof_pred),
                              "Time":      total_time},
                              ignore_index=True)

    return balanced_accuracy_score(y, oof_pred)


study = optuna.create_study(direction="maximize") # TPESampler is used as the default.
study.optimize(objective, n_trials=100, timeout=60) # segs 1hora==60*60
print("Best params:", study.best_trial)
print("Best metric:", study.best_value)


# Save results
optuna_results = study.trials_dataframe()
optuna_results.to_csv(results_directory + 'Optuna_results.csv')



# Visualization
#optuna.visualization.plot_optimization_history(study) # History plot
#optuna.visualization.plot_intermediate_values(study)  # Pruning history
#optuna.visualization.plot_slice(study)                # 1D Params vs metric
#optuna.visualization.plot_contour(study)              # 2D Params vs metric
#optuna.visualization.plot_parallel_coordinate(study)  # ND Params vs metric
#optuna.visualization.plot_param_importances(study)

def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]
     
html = my_results.style.set_precision(8)\
               .bar(subset=metrics, vmin=0, vmax=1, color='rgba(95, 186, 125, 0.8)')\
               .apply(highlight_max, subset=metrics)\
               .set_caption('Models Metrics.')\
               .render()

# SORTABLE
# https://www.kryogenix.org/code/browser/sorttable/
sort_script = "<script src=\"https://www.kryogenix.org/code/browser/sorttable/sorttable.js\"></script>"
html = sort_script + html
html = html.replace("<table", "<table class=\"sortable\"")

# Save
tfile = open("../../results/model_results.html", 'w')
tfile.write(html)
tfile.close()


##################################### MODELS PARAMS



"""
MLPClassifier = {
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "solver" : ["lbfgs", "sgd", "adam"],
    "alpha": np.arange(0, 1, 0.0001),
    "hidden_layer_sizes": [(50,50,50), (50,100,50), (100,), (100,50,100), (100,100,100)],
    "activation": ["tanh", "identity", "logistic", "relu"]
}

GaussianProcessClassifier = {
    "max_iter_predict":[100,200,300,400,500,600,700,800,900,1000]
}

GaussianNB = {
    "var_smoothing": [0.000000001, 0.000000002, 0.000000005, 0.000000008, 0.000000009,
                      0.0000001, 0.0000002, 0.0000003, 0.0000005, 0.0000007, 0.0000009, 
                      0.00001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.009,
                      0.004, 0.005, 0.006, 0.007,0.008, 0.009, 0.01, 0.1, 1]
}

SGDClassifier
linear_svm = {
    "penalty": ["l2", "l1","elasticnet"],
    "l1_ratio": np.arange(0,1,0.01),
    "alpha": [0.0001, 0.001, 0.01, 0.0002, 0.002, 0.02, 0.0005, 0.005, 0.05],
    "fit_intercept": [True, False],
    "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    "eta0": [0.001, 0.01,0.05,0.1,0.2,0.3,0.4,0.5]
}

SVC
rbfsvm = {
    "C": np.arange(0, 50, 0.01),
    "class_weight": ["balanced", None]
}

RidgeClassifier = {
    "alpha": np.arange(0,1,0.001),
    "fit_intercept": [True, False],
    "normalize": [True, False]
}

QuadraticDiscriminantAnalysis = {
    "reg_param": np.arange(0,1,0.01)
}

LinearDiscriminantAnalysis = {
    "solver" : ["lsqr", "eigen"],
    "shrinkage": [None, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
}

DecisionTreeClassifier = {
    "max_depth": np.random.randint(1, (len(X_train.columns)*.85),20),
    "max_features": np.random.randint(1, len(X_train.columns),20),
    "min_samples_leaf": [2,3,4,5,6],
    "criterion": ["gini", "entropy"],
}

RandomForestClassifier = {
    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "criterion": ["gini", "entropy"],
    "max_depth": [int(x) for x in np.linspace(10, 110, num = 11)],
    "min_samples_split": [2, 5, 7, 9, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features" : ["auto", "sqrt", "log2"],
    "bootstrap": [True, False]
}

AdaBoostClassifier = {
    "n_estimators":  np.arange(10,200,5),
    "learning_rate": np.arange(0,1,0.01),
    "algorithm" : ["SAMME", "SAMME.R"]
}

GradientBoostingClassifier = {
    "n_estimators": np.arange(10,200,5),
    "learning_rate": np.arange(0,1,0.01),
    "subsample" : np.arange(0.1,1,0.05),
    "min_samples_split" : [2,4,5,7,9,10],
    "min_samples_leaf" : [1,2,3,4,5],
    "max_depth": [int(x) for x in np.linspace(10, 110, num = 11)],
    "max_features" : ["auto", "sqrt", "log2"]
}

ExtraTreesClassifier = {
    "n_estimators": np.arange(10,200,5),
    "criterion": ["gini", "entropy"],
    "max_depth": [int(x) for x in np.linspace(1, 11, num = 1)],
    "min_samples_split": [2, 5, 7, 9, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features" : ["auto", "sqrt", "log2"],
    "bootstrap": [True, False]
}

XGBClassifier = {
    "learning_rate": np.arange(0,1,0.01),
    "n_estimators":[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
    "subsample": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
    "max_depth": [int(x) for x in np.linspace(10, 110, num = 11)], 
    "colsample_bytree": [0.5, 0.7, 0.9, 1],
    "min_child_weight": [1, 2, 3, 4],
}

LGBMClassifier = {
    "num_leaves":    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
    "max_depth":     [int(x) for x in np.linspace(10, 110, num = 11)],
    "learning_rate": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    "n_estimators":  [10, 30, 50, 70, 90, 100, 120, 150, 170, 200], 
    "min_split_gain": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    "reg_alpha":     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "reg_lambda":    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

CatBoostClassifier = {
    "depth":[3,1,2,6,4,5,7,8,9,10],
    "iterations":[250,100,500,1000], 
    "learning_rate":[0.03,0.001,0.01,0.1,0.2,0.3], 
    "l2_leaf_reg":[3,1,5,10,100], 
    "border_count":[32,5,10,20,50,100,200], 
}

"""