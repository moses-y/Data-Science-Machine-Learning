#!pip install optuna
#!pip install umap-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.utils import to_categorical
from keras.models import Model
import optuna
from sklearn.metrics import log_loss

# Load data
data = pd.read_csv(r"C:\Users\moses_y\OneDrive\Desktop\ML Projects\Log Loss\train_data_swc.csv").dropna()

# Feature engineering
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
polynomial_features = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = polynomial_features.fit_transform(X)

# Dimensionality reduction
pca = PCA(n_components=0.95)  # Maintain 95% of variance
X_pca = pca.fit_transform(X_poly)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_pca, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

# Categorical encoding
y_train_enc = to_categorical(y_train)
y_valid_enc = to_categorical(y_valid)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Define the objective function for hyperparameter tuning
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-5, 1e-2)

    model = Sequential()
    model.add(Dense(trial.suggest_int('n_units_l0', 50, 300), activation='relu', input_dim=X_train_scaled.shape[1],
                    kernel_regularizer=l1_l2(l2=reg_lambda)))
    model.add(Dropout(dropout_rate))
    for i in range(n_layers):
        model.add(Dense(trial.suggest_int(f'n_units_l{i+1}', 50, 300), activation='relu',
                        kernel_regularizer=l1_l2(l2=reg_lambda)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(y_train_enc.shape[1], activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train_enc, epochs=50, batch_size=32, validation_data=(X_valid_scaled, y_valid_enc), verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_valid_scaled)
    return log_loss(y_valid_enc, y_pred)

# Optimize with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params

# Build final model
final_model = Sequential([
    Dense(best_params['n_units_l0'], activation='relu', input_dim=X_train_scaled.shape[1], kernel_regularizer=l1_l2(l2=best_params['reg_lambda'])),
    Dropout(best_params['dropout_rate'])
])
for i in range(best_params['n_layers']):
    final_model.add(Dense(best_params[f'n_units_l{i+1}'], activation='relu', kernel_regularizer=l1_l2(l2=best_params['reg_lambda'])))
    final_model.add(Dropout(best_params['dropout_rate']))
final_model.add(Dense(y_train_enc.shape[1], activation='softmax'))

final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])
final_model.fit(X_train_scaled, y_train_enc, epochs=100, batch_size=32, validation_data=(X_valid_scaled, y_valid_enc), verbose=1)

# Save the model if needed
# final_model.save('final_model.h5')
