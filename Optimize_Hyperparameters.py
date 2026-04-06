
"""
Hyperparameter Optimization for Stacking Ensemble Model
=======================================================
This script uses Optuna framework to find optimal hyperparameters for the stacking ensemble model defined in the main Jupyter notebook.
The best parameters found will be printed upon completion. These parameters can then be manually updated in the main notebook.
WARNING: This script is computationally intensive and may take a significant amount of time to run, depending on the number of trials (`n_trials`).

Requirements:
    - DATASET.xlsx 
    - Required packages: 
       pip install optuna  
       pip install pandas 
       pip install numpy
"""

import optuna
import numpy as np
import pandas as pd
# Scikit-learn imports
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor


# Load and prepare data 
df = pd.read_excel('DATASET.xlsx')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

features = ["IOLtype", "ACD", "LT", "AL", "K1", "K2", "WTW", "IOL"]
target = "SEQ"

X = df[features].copy()
y = df[target].copy()

# Encode categorical feature
le = LabelEncoder()
X['IOLtype'] = le.fit_transform(X['IOLtype'])

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
cols_to_scale = ["ACD", "LT", "AL", "K1", "K2", "WTW", "IOL"]
X_train_scaled = X_train.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])




# Optuna objective function for hyperparameter optimization
def objective(trial):
    
    # MLP parameters
    mlp_hidden = trial.suggest_categorical("mlp_hidden_layer_sizes", 
                                          [(50,), (100,), (200,), (100,50)])
    mlp_alpha = trial.suggest_loguniform("mlp_alpha", 1e-5, 1e-1)
    mlp_lr = trial.suggest_loguniform("mlp_lr", 1e-4, 1e-2)
    mlp_activation = trial.suggest_categorical("mlp_activation", ['relu', 'tanh'])
    mlp_max_iter = trial.suggest_int("mlp_max_iter", 500, 1000, step=100)

    # SVR parameters
    svr_C = trial.suggest_loguniform("svr_C", 1e-2, 1e3)
    svr_epsilon = trial.suggest_loguniform("svr_epsilon", 1e-3, 1.0)
    svr_kernel = trial.suggest_categorical("svr_kernel", ["linear", "rbf"])
    svr_gamma = trial.suggest_categorical("svr_gamma", ['scale', 'auto', 0.01, 0.03, 0.1])

    # Spline parameters
    spline_degree = trial.suggest_int("spline_degree", 2, 4)
    spline_knots = trial.suggest_int("spline_n_knots", 5, 8)

    # Ridge parameters
    ridge_alpha = trial.suggest_loguniform("ridge_alpha", 1e-3, 10)

    # Build base learners
    base_learners = [
        ('mlp', MLPRegressor(
            hidden_layer_sizes=mlp_hidden,
            alpha=mlp_alpha,
            learning_rate_init=mlp_lr,
            activation=mlp_activation,
            max_iter=mlp_max_iter,
            random_state=42
        )),
        ('spline_linear', Pipeline([
            ('spline', SplineTransformer(
                degree=spline_degree, 
                n_knots=spline_knots
            )),
            ('linear', LinearRegression())
        ])),
        ('svr', SVR(
            C=svr_C,
            epsilon=svr_epsilon,
            kernel=svr_kernel,
            gamma=svr_gamma
        ))
    ]

    # Meta learner
    meta_learner = Ridge(alpha=ridge_alpha, random_state=42)

    # Stacking Regressor
    model = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train_scaled, y_train,
                            cv=cv, scoring="neg_mean_squared_error").mean()
    return -score  # minimize MSE

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best trial:", study.best_trial.params)