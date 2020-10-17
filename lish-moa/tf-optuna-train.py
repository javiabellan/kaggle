import numpy as np
import pandas as pd
import joblib       # To save & load preprocesing
import time
from IPython.display import display, clear_output


#################################################### SKLEARN UTILS
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

#################################################### ML MODELS
#from sklearn.neighbors      import KNeighborsClassifier
#from sklearn.linear_model   import LogisticRegression, Ridge, RidgeClassifier
#from sklearn.svm            import LinearSVC
#from sklearn.neural_network import MLPClassifier
#from sklearn.naive_bayes   import MultinomialNB
#from sklearn.linear_model  import SGDClassifier
#from sklearn.multiclass     import OneVsRestClassifier
#from sklearn.multioutput    import MultiOutputClassifier

############################################# TENSORFLOW
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow_addons as tfa
#from keras_drop_connect import DropConnect
#from keras_lr_finder import LRFinder

import optuna



########################## My code

# from application.app.folder.file import func_name
from utils.OneCycle import OneCycleScheduler


oc = OneCycleScheduler(0.1, 10)

print("Tensorflow:", tf.__version__)

# Deep Learning
#import fastai
#print("Fast.ai:",fastai.__version__)


#if tf.test.is_gpu_available(): print('Found GPU at:', tf.test.gpu_device_name())
#else: print("GPU device not found")
#gpu = tf.config.list_physical_devices('GPU')
#print(gpu)

############################################################## DATA


path = "../../data-moa/"

x   = pd.read_csv(path+"train_features.csv")
y   = pd.read_csv(path+"train_targets_scored.csv")
y2  = pd.read_csv(path+"train_targets_nonscored.csv")
x_t = pd.read_csv(path+"test_features.csv")
sub = pd.read_csv(path+"sample_submission.csv")

y  = y.drop('sig_id',axis=1)
y2 = y2.drop('sig_id',axis=1)

genes_vars = [col for col in x.columns if col.startswith('g-')]
cells_vars = [col for col in x.columns if col.startswith('c-')]

print("Train")
print("x:", x.shape, "GENES:", len(genes_vars), "CELLS:", len(cells_vars))
print("y:", y.shape)
print("y2:", y2.shape)
print("Test")
print("x:", x_t.shape)



############################################################## PREPROCESING

cat_vars = ["cp_dose", "cp_type", "cp_time"]
num_vars  = genes_vars + cells_vars

cats_oh_vars = ['type1', 'type2', 'time1', 'time2', 'time3', 'dose1', "dose2"]
prepro_cols = cats_oh_vars + genes_vars + cells_vars

preprocessor = ColumnTransformer([
   # ('ordinal',    OrdinalEncoder(), num_feats),
    ('onehot', OneHotEncoder(), cat_vars),
    ("scale", StandardScaler(), num_vars)
])

joblib.dump(preprocessor, 'ColumnTransformer.joblib')



############################################################## MODEL

def NN2(gene_dim, cell_dim, hid1_dim, hid2_dim, act, dropout):

    cats_feats = K.Input(shape=(7,),   name="CETEGORIES")
    gene_feats = K.Input(shape=(772,), name="GENES")
    cell_feats = K.Input(shape=(100,), name="CELLS")
    
    gene_emb = K.layers.Dropout(dropout)(gene_feats)
    gene_emb = K.layers.BatchNormalization()(gene_emb)
    gene_emb = tfa.layers.WeightNormalization(K.layers.Dense(gene_dim, activation=act))(gene_emb)
 
    cell_emb = K.layers.Dropout(dropout)(cell_feats)
    cell_emb = K.layers.BatchNormalization()(cell_emb)
    cell_emb = tfa.layers.WeightNormalization(K.layers.Dense(cell_dim, activation=act))(cell_emb)

    all_feats = K.layers.Concatenate(name="ALL")([cats_feats, gene_feats, cell_feats, gene_emb, cell_emb])

    #hid1 = DropConnect(layer=K.layers.Dense(hid1_dim, activation=act), rate=dropconnect)(all_feats)
    hid1 = tfa.layers.WeightNormalization(K.layers.Dense(hid1_dim, activation=act))(all_feats)
    out = K.layers.Dropout(dropout)(hid1)
    out = K.layers.BatchNormalization()(out)
    
    hid2 = tfa.layers.WeightNormalization(K.layers.Dense(hid2_dim, activation=act))(out)
    out = K.layers.Dropout(dropout)(hid2)
    out = K.layers.BatchNormalization()(out)
    
    scored_y    = tfa.layers.WeightNormalization(K.layers.Dense(206, activation='sigmoid'))(out)
    nonscored_y = tfa.layers.WeightNormalization(K.layers.Dense(402, activation='sigmoid'))(out)
    
    return K.Model(
        inputs=[cats_feats, gene_feats, cell_feats],
        outputs=[scored_y, nonscored_y],
    )




############################################################## HYPERPARAMS OPT


def objective(trial):
    global results
    global plots

    ######## DataAug params
    gauss_noise = 0.1
    cutmix      = False
    mixup       = False
    std_scaler  = False # StandardScaler fon num feats from sklearn
    #smote ???

    ######## Model params
    #hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    gene_dim      = trial.suggest_int('gene_dim', 10, 300, 10)
    cell_dim      = trial.suggest_int('cell_dim', 5, 50, 5)
    hid1_dim      = trial.suggest_int('hid1_dim', 500, 4000, 10)
    hid2_dim      = trial.suggest_int('hid2_dim', 300, 2000, 10)
    activation    = trial.suggest_categorical('activation', ["relu", "elu"]) # swish, mish
    dropout       = trial.suggest_discrete_uniform('dropout', 0.05, 0.7, 0.05)
    dropconnect   = False
    #VAE???

	######## Loss params
    scored_pct = trial.suggest_loguniform('nonscored_pct', 0.5, 1.0) # To inclue non_scored target
    label_smooth  = # Label smoothing means increasing uncertainty.
    clip_preds = 0.001 # y_pred = tf.clip_by_value(y_pred, p_min=clip_preds, p_max=1-clip_preds)
    # https://www.kaggle.com/rahulsd91/moa-label-smoothing

	########## Train params    
    epochs        = trial.suggest_int('epochs', 1, 50)
    batch_size    = trial.suggest_int('batch_size', low=64, high=192, step=32)
    lr_max        = trial.suggest_loguniform('lr', 0.001, 0.1)
    lr_sched      = "OneCycle"
    optimizer     = "adam"
    lookahead     = trial.suggest_categorical('lookahead', [False, True])
    
    """
    model = NN(hidden_layers=hidden_layers,
               hidden_dim=hidden_dim,
               act=activation,
               dropout=dropout,
               lr_max=lr_max,
               lookahead=lookahead)
    """
    model = NN2(gene_dim=gene_dim,
                cell_dim=cell_dim,
                hid1_dim=hid1_dim,
                hid2_dim=hid2_dim,
                act=activation,
                dropout=dropout)
    
    
    if optimizer=="adam":  
    	optimizer = K.optimizers.Adam(learning_rate=lr_max)
    if lookahead:
    	optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=10)


    l1 = 0.7
    l2 = 0.3
    model.compile(optimizer=optimizer,
                  loss = ['binary_crossentropy','binary_crossentropy'], # loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
                  loss_weights = [scored_pct, 1-scored_pct])

    loss_scored = 999999
    for fold_i, (tr_idx, vl_idx) in enumerate(CV.split(x, y)):

        #print('Fold', fold_i)

        ############### PREPRO
        x_train = preprocessor.fit_transform(x.iloc[tr_idx]); x_train = pd.DataFrame(x_train, columns=prepro_cols)
        x_valid = preprocessor.transform(x.iloc[vl_idx]);     x_valid = pd.DataFrame(x_valid, columns=prepro_cols)
        y_train = y.iloc[tr_idx]#.values
        y_valid = y.iloc[vl_idx]#.values
        y2_train = y2.iloc[tr_idx]#.values
        y2_valid = y2.iloc[vl_idx]#.values
        
        callbacks = []
        if lr_sched=="RedOnPla":
            #callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')) # epsilon=1e-4
            callbacks.append(ReduceLROnPlateau(factor=0.5, patience=4, verbose=0, mode="auto"))
        if lr_sched=="OneCycle":
            steps = np.ceil(len(x_train) / batch_size) * epochs
            callbacks.append(OneCycleScheduler(lr_max, steps))
        
        ############### FIT
        start_time = time.time()
        hist = model.fit(
                  x = [x_train[cats_oh_vars], x_train[genes_vars], x_train[cells_vars]],
                  y = [y_train, y2_train],
                  validation_data=([x_valid[cats_oh_vars], x_valid[genes_vars], x_valid[cells_vars]], [y_valid, y2_valid]),
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=callbacks) #  verbose=1
        fit_time = time.time() - start_time
        
        plots.append(hist.history)

        #model.load_weights(checkpoint_path)
        #val_predict = model.predict(x_valid)
        
        loss, loss_scored, loss_nonscored = model.evaluate([x_valid[cats_oh_vars], x_valid[genes_vars], x_valid[cells_vars]], [y_valid, y2_valid])
        score = loss_scored
        #score = model.evaluate(x_valid, y_valid, return_dict=True)["BCE_loss"] #BUG: https://github.com/keras-team/keras-io/issues/232
        #results.at[index, 'Fold'+str(fold_i)] = score
        """
        results = results.append({
             'hidden_layers': hidden_layers,
             'hidden_dim': hidden_dim,
             'batch_size': batch_size,
             'dropout': dropout,
             'activation': activation,
             'epochs': epochs,
             'lookahead': lookahead,
             'lr_max': lr_max,
             'lr_sched': lr_sched,
             'score': score,
             'time': fit_time
        },ignore_index=True)
        """
        results = results.append({
             'gene_dim': gene_dim,
             'cell_dim': cell_dim,
             'hid1_dim': hid1_dim,
             'hid2_dim': hid2_dim,
             'batch_size': batch_size,
             'dropout': dropout,
             'activation': activation,
             'epochs': epochs,
             'lookahead': lookahead,
             'lr_max': lr_max,
             #'lr_sched': lr_sched,
             'score': score,
             'time': fit_time
        },ignore_index=True)

        #clear_output()
        #display(results) # OR print(results.to_html())
        results.to_csv('my_results.csv')
        break

    return score




RANDOM_TRIALS = 50   # 25
TOTAL_TRIALS  = 1000 # 100
SEED          = 17
FOLDS         = 3

np.random.seed(SEED)
tf.random.set_seed(SEED)

results = pd.DataFrame()
CV = MultilabelStratifiedKFold(n_splits=FOLDS, random_state=SEED, shuffle=True)
plots = []
    
#optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
             direction="minimize",
             sampler=optuna.samplers.TPESampler(n_startup_trials=RANDOM_TRIALS))

study.optimize(objective, n_trials=TOTAL_TRIALS) # timeout=60 # segs

# optuna.importance.get_param_importances


# Save results
#print("Best params:", study.best_trial)
#print("Best metric:", study.best_value)
#df_results = study.trials_dataframe()
#df_results.to_csv('df_optuna_results.csv')