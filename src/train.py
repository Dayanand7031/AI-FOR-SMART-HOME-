import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ML Models
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# DL Models (with fallback for missing tensorflow)
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, GRU, Conv1D, MaxPooling1D, Flatten, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Deep Learning models will be skipped.")

# Validation and Tuning
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge

from config import BEST_MODEL_PATH, RANDOM_STATE, SEQUENCE_LENGTH

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def build_lstm(input_shape):
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(100, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(100, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_ml_models(X_train, y_train, X_val, y_val, use_tuning=False):
    """
    Train ML models with optional hyperparameter tuning for Random Forest, XGBoost, and LightGBM.
    When use_tuning=True, uses systematic hyperparameter tuning with GridSearchCV and TimeSeriesSplit
    to find optimal parameters without data leakage.
    """
    # Initialize models
    if use_tuning:
        # Use systematic hyperparameter tuning with GridSearchCV
        print("Using systematic hyperparameter tuning with GridSearchCV...")

        # Define parameter grids for each model
        # These ranges are chosen based on typical effective values and our previous experiments
        rf_param_grid = {
            'n_estimators': [200, 300, 400],  # Number of trees in the forest
            'max_depth': [10, 15, 20, None],  # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required to be at a leaf node
        }

        xgb_param_grid = {
            'n_estimators': [300, 400, 500],  # Number of gradient boosted trees
            'learning_rate': [0.01, 0.03, 0.05],  # Step size shrinkage used in update to prevents overfitting
            'max_depth': [4, 6, 8],  # Maximum depth of a tree
            'subsample': [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
            'colsample_bytree': [0.6, 0.8, 1.0]  # Subsample ratio of columns when constructing each tree
        }

        lgbm_param_grid = {
            'n_estimators': [300, 400, 500],  # Number of boosted trees
            'learning_rate': [0.01, 0.03, 0.05],  # Boosting learning rate
            'max_depth': [4, 6, 8],  # Maximum depth of the tree
            'min_child_samples': [10, 20, 30],  # Minimum number of data needed in a child
            'subsample': [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
            'colsample_bytree': [0.6, 0.8, 1.0]  # Subsample ratio of columns when constructing each tree
        }

        # Use TimeSeriesSplit for cross-validation to prevent data leakage
        tss = TimeSeriesSplit(n_splits=3)

        # Tune Random Forest
        print("Tuning Random Forest...")
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE),
            rf_param_grid,
            cv=tss,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        rf_model = rf_grid.best_estimator_
        print(f"Best RF parameters: {rf_grid.best_params_}")

        # Tune XGBoost
        print("Tuning XGBoost...")
        xgb_grid = GridSearchCV(
            XGBRegressor(random_state=RANDOM_STATE),
            xgb_param_grid,
            cv=tss,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        xgb_model = xgb_grid.best_estimator_
        print(f"Best XGBoost parameters: {xgb_grid.best_params_}")

        # Tune LightGBM
        print("Tuning LightGBM...")
        lgbm_grid = GridSearchCV(
            LGBMRegressor(random_state=RANDOM_STATE),
            lgbm_param_grid,
            cv=tss,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        lgbm_grid.fit(X_train, y_train)
        lgbm_model = lgbm_grid.best_estimator_
        print(f"Best LightGBM parameters: {lgbm_grid.best_params_}")
    else:
        # Default parameters
        rf_model = RandomForestRegressor(random_state=RANDOM_STATE)
        xgb_model = XGBRegressor(random_state=RANDOM_STATE)
        lgbm_model = LGBMRegressor(random_state=RANDOM_STATE)

    # Additional XGBoost with Huber loss for robustness to outliers
    xgb_huber_model = XGBRegressor(
        random_state=RANDOM_STATE,
        objective='reg:huberloss',  # More robust to outliers than default squared error
        alpha=0.9  # Huber parameter: smaller values = more robust to outliers
    )

    models = {
        "random_forest": rf_model,
        "xgboost": xgb_model,
        "lightgbm": lgbm_model,
        "xgboost_huber": xgb_huber_model
    }

    results = {}
    for name, model in models.items():
        # For tuned models, they are already fitted from grid search
        # For non-tuned models (like our Huber XGBoost), we need to fit them
        if name in ["random_forest", "xgboost", "lightgbm"] and use_tuning:
            # These models were already fitted during grid search
            pass
        else:
            # Fit the model (needed for non-tuned models or when not tuning)
            model.fit(X_train, y_train)

        preds = model.predict(X_val)

        results[name] = {
            "rmse": np.sqrt(mean_squared_error(y_val, preds)),
            "mae": mean_absolute_error(y_val, preds),
            "mape": calculate_mape(y_val, preds),
            "r2": r2_score(y_val, preds),
            "model": model
        }
    return results

def train_dl_models(X_train, y_train, X_val, y_val):
    # Return empty results if TensorFlow is not available
    if not TF_AVAILABLE:
        return {}

    # DL models require 3D input (samples, time_steps, features)
    def create_sequences(X, y):
        Xs, ys = [], []
        for i in range(len(X) - SEQUENCE_LENGTH):
            Xs.append(X[i:(i + SEQUENCE_LENGTH)])
            ys.append(y.iloc[i + SEQUENCE_LENGTH])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val)

    models_dict = {
        "lstm": build_lstm((SEQUENCE_LENGTH, X_train.shape[1])),
        "gru": build_gru((SEQUENCE_LENGTH, X_train.shape[1])),
        "cnn_lstm": build_cnn_lstm((SEQUENCE_LENGTH, X_train.shape[1]))
    }

    results = {}
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    for name, model in models_dict.items():
        model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq),
                  epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

        preds = model.predict(X_val_seq).flatten()

        results[name] = {
            "rmse": np.sqrt(mean_squared_error(y_val_seq, preds)),
            "mae": mean_absolute_error(y_val_seq, preds),
            "mape": calculate_mape(y_val_seq, preds),
            "r2": r2_score(y_val_seq, preds),
            "model": model
        }
    return results

def train_ensemble_models(X_train, y_train, X_val, y_val, base_models):
    """
    Train ensemble models (Voting and Stacking) using base models.
    """
    # Use the base models from ML (with tuning if available)
    if base_models:
        # Use tuned models if available
        rf = base_models["random_forest"]["model"] if "random_forest" in base_models else RandomForestRegressor(random_state=RANDOM_STATE)
        xgb = base_models["xgboost"]["model"] if "xgboost" in base_models else XGBRegressor(random_state=RANDOM_STATE)
        lgbm = base_models["lightgbm"]["model"] if "lightgbm" in base_models else LGBMRegressor(random_state=RANDOM_STATE)
        # Include Huber XGBSD if available
        xgb_huber = base_models["xgboost_huber"]["model"] if "xgboost_huber" in base_models else None
    else:
        # Fallback to default parameters
        rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        xgb = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE)
        lgbm = LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE)
        xgb_huber = None

    # Prepare base models for ensemble
    base_model_list = [('rf', rf), ('xgb', xgb), ('lgbm', lgbm)]
    if xgb_huber is not None:
        base_model_list.append(('xgb_huber', xgb_huber))

    # Voting Regressor
    voting = VotingRegressor(base_model_list)

    # Stacking Regressor - use the original models (without Huber) for consistency
    stacking = StackingRegressor(
        estimators=[
            ('rf', rf),
            ('xgb', xgb)
        ],
        final_estimator=Ridge(),
        cv=3
    )

    ensemble_models = {
        "voting": voting,
        "stacking": stacking
    }

    results = {}
    for name, model in ensemble_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        results[name] = {
            "rmse": np.sqrt(mean_squared_error(y_val, preds)),
            "mae": mean_absolute_error(y_val, preds),
            "mape": calculate_mape(y_val, preds),
            "r2": r2_score(y_val, preds),
            "model": model
        }
    return results

def train_and_evaluate():
    from src.data_ingestion import fetch_nasa_data
    from src.preprocessing import preprocess_data

    # 1. Data Prep
    df_raw = fetch_nasa_data()
    X_train_scaled, X_test_scaled, y_train, y_test, _ = preprocess_data(df_raw, remove_outliers=False)

    # Convert to numpy for consistency
    X_train = X_train_scaled
    X_test = X_test_scaled

    # Use TimeSeriesSplit for cross-validation logic (we'll use 3 splits for tuning/validation)
    tss = TimeSeriesSplit(n_splits=3)
    # We'll use the last split for validation in this simplified approach
    for train_index, val_index in tss.split(X_train):
        curr_X_train, curr_X_val = X_train[train_index], X_train[val_index]
        curr_y_train, curr_y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # 2. Train ML Models (with hyperparameter tuning for RF)
    print("Training ML Models...")
    ml_results = train_ml_models(curr_X_train, curr_y_train, curr_X_val, curr_y_val, use_tuning=True)

    # 3. Train DL Models (if TensorFlow is available)
    print("Training DL Models...")
    dl_results = {}
    if TF_AVAILABLE:
        dl_results = train_dl_models(curr_X_train, curr_y_train, curr_X_val, curr_y_val)
    else:
        print("Skipping Deep Learning models due to missing TensorFlow.")

    # 4. Train Ensemble Models
    print("Training Ensemble Models...")
    # Use the same train/val split for ensemble training
    ensemble_results = train_ensemble_models(curr_X_train, curr_y_train, curr_X_val, curr_y_val, ml_results)

    # Combine all results
    all_results = {**ml_results, **dl_results, **ensemble_results}
    if not all_results:
        raise RuntimeError("No models were trained. Check if ML dependencies are installed.")

    # Find best model based on RMSE
    best_model_name = min(all_results, key=lambda k: all_results[k]["rmse"])
    best_model = all_results[best_model_name]["model"]

    # 5. Save Best Model
    joblib.dump(best_model, BEST_MODEL_PATH)

    # 6. Log Results
    log_data = []
    for name, metrics in all_results.items():
        log_data.append([name, metrics["rmse"], metrics["mae"], metrics["mape"], metrics["r2"]])

    log_df = pd.DataFrame(log_data, columns=["Model", "RMSE", "MAE", "MAPE", "R2"])
    log_df.to_csv("training_log.csv", index=False)

    print(f"Training Complete. Best Model: {best_model_name} with RMSE: {all_results[best_model_name]['rmse']:.4f}")
    print("\nFull Results:")
    print(log_df.to_string(index=False))
    return best_model_name, all_results

if __name__ == "__main__":
    train_and_evaluate()