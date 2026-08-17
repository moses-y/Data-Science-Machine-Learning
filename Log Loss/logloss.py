
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier, plot_importance
from keras.models import Sequential
from keras.layers import Dense, GaussianNoise, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1
from pytorch_tabnet.tab_model import TabNetClassifier
import torch  # Importing torch here
import optuna
from optuna.integration import LightGBMPruningCallback

import warnings
warnings.filterwarnings("ignore")

# Constants
AE_EPOCHS, NN_EPOCHS, LG_MAX_ITER, TN_MAX_EPOCHS, OPTUNA_N_TRIALS = 500, 500, 500, 500, 30

# Helper functions
def calculate_vif(X, thresh=100.0):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in range(X.iloc[:, variables].shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(f'Dropping {X.columns[variables[maxloc]]} with VIF {max(vif)}')
            del variables[maxloc]
            dropped = True
    return X.iloc[:, variables]

def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(df[to_drop], axis=1), to_drop

# Data preprocessing
train_data = pd.read_csv(r"C:\Users\moses_y\OneDrive\Desktop\ML Projects\Log Loss\train_data_swc.csv").dropna()
train_data, to_drop = remove_highly_correlated_features(train_data)

X = train_data.drop('y', axis=1)
y = train_data['y']

# Feature engineering
X = calculate_vif(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y-1, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Modeling
rf = RandomForestClassifier(n_estimators=150, random_state=42)
et = ExtraTreesClassifier(random_state=42)
kn = KNeighborsClassifier()
lg = LogisticRegression(max_iter=LG_MAX_ITER, random_state=42)
tn = TabNetClassifier(optimizer_params=dict(lr=1e-4), scheduler_params={"step_size":10, "gamma":0.3}, scheduler_fn=torch.optim.lr_scheduler.StepLR, verbose=0)

nn = Sequential([
    GaussianNoise(0.01),
    Dense(300, activation='elu', kernel_regularizer=l1(0.0002)),
    Dropout(0.3),
    Dense(200, activation='elu', kernel_regularizer=l1(0.0002)),
    Dropout(0.3),
    Dense(100, activation='elu', kernel_regularizer=l1(0.0002)),
    Dropout(0.3),
    Dense(50, activation='sigmoid', kernel_regularizer=l1(0.002)),
    Dropout(0.2),
    Dense(len(np.unique(y)), activation='softmax')
])
nn.compile(optimizer=Adam(0.00011), loss='sparse_categorical_crossentropy')

# Model training
nn.fit(X_train, y_train, epochs=NN_EPOCHS, batch_size=32, validation_data=(X_valid, y_valid))

models = [rf, et, kn, lg, tn]
for model in models:
    model.fit(X_train, y_train)
    print(f"{model.__class__.__name__} log loss: {log_loss(y_test, model.predict_proba(X_test))}")

# Optuna for hyperparameter tuning
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': len(np.unique(y)),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'verbose': -1
    }
    model = LGBMClassifier(**params)
    return cross_val_score(model, X_train, y_train, n_jobs=-1, cv=KFold(n_splits=3), scoring='neg_log_loss').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

params = study.best_params
model = LGBMClassifier(**params)
model.fit(X_train, y_train)
