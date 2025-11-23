from pathlib import Path
from src.data_preprocessing import load_and_preprocess_data
from src.model_build import build_dnn_model
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle


def train_model(data_path: str = None):
    # default dataset path relative to repo root
    if data_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / 'data' / 'Phising_Detection_Dataset.csv'

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(str(data_path))

    model = build_dnn_model(input_dim=X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/phishing_model.h5")
    # Save scaler so inference can use the same normalization
    try:
        with open(os.path.join('models', 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ Scaler saved at models/scaler.pkl")
    except Exception:
        print("⚠️ Could not save scaler to models/scaler.pkl")
    print("✅ Model saved at models/phishing_model.h5")

    return model, X_test, y_test, history
