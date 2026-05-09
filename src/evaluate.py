import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from config import BEST_MODEL_PATH, SCALER_PATH, OUTPUTS_DIR, PROCESSED_DATA_DIR

def evaluate_model():
    """
    Evaluates the best saved model on the test set and generates performance plots.
    """
    # 1. Load data and model
    from src.data_ingestion import fetch_nasa_data
    from src.preprocessing import preprocess_data
    import joblib
    from config import SCALER_PATH, PROCESSED_DATA_DIR, TRAIN_TEST_SPLIT

    df_raw = fetch_nasa_data()

    # Load the scaler that was saved during training
    scaler = joblib.load(SCALER_PATH)

    # Load the processed data that was saved during training
    processed_df = pd.read_csv(PROCESSED_DATA_DIR / "processed_energy_data.csv")

    # Now we need to split this into train/test the same way as during training
    split_idx = int(len(processed_df) * TRAIN_TEST_SPLIT)

    # Features are all columns except date and target
    # We also remove ALLSKY_SFC_SW_DWN because it's the target for the next day
    # (though the lag features already contain its info)
    feature_cols = [col for col in processed_df.columns if col not in ['date', 'target', 'ALLSKY_SFC_SW_DWN']]
    X = processed_df[feature_cols]
    y = processed_df['target']

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Now scale using the saved scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Load the model
    model = joblib.load(BEST_MODEL_PATH)

    # Handle DL model sequence requirements if necessary
    # We check if the model is a Keras/TF model
    is_keras_model = hasattr(model, 'predict') and hasattr(model, 'layers')

    if is_keras_model:
        # Create sequences for test set
        def create_sequences(X, y):
            Xs, ys = [], []
            for i in range(len(X) - 7):
                Xs.append(X[i:(i + 7)])
                ys.append(y.iloc[i + 7])
            return np.array(Xs), np.array(ys)

        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)
        preds = model.predict(X_test_seq).flatten()
        actuals = y_test_seq
    else:
        preds = model.predict(X_test_scaled).flatten()
        actuals = y_test.values

    # 2. Metrics Calculation
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    print(f"Debug: actuals shape: {actuals.shape}, preds shape: {preds.shape}")
    print(f"Debug: actuals min: {np.min(actuals)}, max: {np.max(actuals)}")
    print(f"Debug: preds min: {np.min(preds)}, max: {np.max(preds)}")
    print(f"Debug: actuals head: {actuals[:5]}")
    print(f"Debug: preds head: {preds[:5]}")
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)

    print(f"--- Final Evaluation Metrics ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # 3. Plotting
    plt.figure(figsize=(15, 6))

    # Plot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.plot(actuals, label='Actual', color='blue', alpha=0.7)
    plt.plot(preds, label='Predicted', color='red', linestyle='--', alpha=0.7)
    plt.title("Actual vs Predicted Energy Consumption")
    plt.xlabel("Days")
    plt.ylabel("SFC SW DWN (kWh/m²/day)")
    plt.legend()

    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    residuals = actuals - preds
    sns.histplot(residuals, kde=True, color='green')
    plt.title("Residual Distribution")
    plt.xlabel("Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "evaluation_plots.png")
    plt.close()

    # 4. Feature Importance (For Tree-based models)
    if not is_keras_model and hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feat_importances = model.feature_importances_
        # Get feature names from the processed CSV
        processed_df = pd.read_csv(PROCESSED_DATA_DIR / "processed_energy_data.csv")
        features = [col for col in processed_df.columns if col not in ['date', 'target']]

        sorted_idx = np.argsort(feat_importances)[::-1]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_importances = feat_importances[sorted_idx]

        plt.barh(sorted_features, sorted_importances, color='skyblue')
        plt.title("Feature Importance")
        plt.xlabel("Score")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "feature_importance.png")
        plt.close()

    print(f"Plots saved to {OUTPUTS_DIR}")

if __name__ == "__main__":
    evaluate_model()
