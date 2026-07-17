import time
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, Multiply
from tensorflow.keras.models import Model

from model_eval import calculate_mae, calculate_rmse, calculate_r2, get_final_losses, plot_loss

DATA_PATH = Path("mixed_dataset_balanced_16k(123456).pkl")
TEST_SPLIT_PATH = Path("mixed_dataset_test_split.pkl")
EPOCHS = 1000
BATCH_SIZE = 32
LR = 1e-3
TEST_SIZE = 0.1
VALID_SIZE = 0.2
SCALER_PATH = Path("SS_scaler_mixed_masked_16k(123456).pkl")
MODEL_WEIGHTS = Path("mixed_masked_model_16k(123456).weights.h5")
MODEL_PATH = Path("mixed_masked_model_16k(123456).h5")


def masked_mse(out_dim):
    def loss(y_true, y_pred):
        y = y_true[:, :out_dim]
        mask = y_true[:, out_dim:]
        mask = tf.cast(mask, y_pred.dtype)
        diff = (y_pred - y) * mask
        mse = tf.reduce_sum(tf.square(diff), axis=1)
        denom = tf.reduce_sum(mask, axis=1) + tf.keras.backend.epsilon()
        return tf.reduce_mean(mse / denom)

    return loss


def build_model(input_shape, out_dim):
    feats = Input(shape=input_shape, name="features")
    mask_in = Input(shape=(out_dim,), name="mask")
    x = Conv1D(32, 3, activation="relu")(feats)
    x = Conv1D(64, 3, activation="relu")(x)
    x = Conv1D(128, 3, activation="relu")(x)
    x = Conv1D(256, 3, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    raw_out = Dense(out_dim)(x)
    masked_out = Multiply()([raw_out, mask_in])
    model = Model([feats, mask_in], masked_out)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=masked_mse(out_dim))
    return model


def main():
    with DATA_PATH.open("rb") as f:
        raw_df = pickle.load(f)
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError(f"{DATA_PATH} must be a pandas DataFrame.")

    df = raw_df.copy()
    if "label" not in df.columns or "H" not in df.columns or "mask" not in df.columns:
        raise ValueError("Dataset must contain 'H', 'label', and 'mask' columns.")

    df["H_vector"] = df["H"].apply(lambda m: np.asarray(m).flatten(order="C"))
    df["W_vector"] = df["label"].apply(lambda m: np.asarray(m).flatten(order="C"))
    df["mask_vector"] = df.apply(
        lambda row: np.tile(np.asarray(row["mask"], dtype=float), np.asarray(row["H"]).shape[0]),
        axis=1,
    )
    df["input_vec"] = df.apply(lambda row: np.concatenate([row["H_vector"], [row["power"]]]), axis=1)

    X = np.stack(df["input_vec"].values)
    y = np.stack(df["W_vector"].values)
    masks = np.stack(df["mask_vector"].values)
    indices = np.arange(len(df))

    X_train_val, X_test, y_train_val, y_test, m_train_val, m_test, idx_train_val, idx_test = train_test_split(
        X, y, masks, indices, test_size=TEST_SIZE, random_state=42, shuffle=True
    )
    valid_frac = VALID_SIZE / (1 - TEST_SIZE)
    X_train, X_valid, y_train, y_valid, m_train, m_valid = train_test_split(
        X_train_val,
        y_train_val,
        m_train_val,
        test_size=valid_frac,
        random_state=42,
        shuffle=True,

    )

    test_df = df.iloc[idx_test].copy()
    test_df.to_pickle(TEST_SPLIT_PATH)
    print(f"Saved test split with {len(test_df)} rows to {TEST_SPLIT_PATH}")


    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, SCALER_PATH)

    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    X_train_r = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_valid_r = X_valid_scaled.reshape(X_valid_scaled.shape[0], X_valid_scaled.shape[1], 1)
    X_test_r = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    out_dim = y.shape[1]
    y_train_packed = np.concatenate([y_train, m_train], axis=1)
    y_valid_packed = np.concatenate([y_valid, m_valid], axis=1)

    model = build_model(X_train_r.shape[1:], out_dim)
    model.summary()

    if MODEL_WEIGHTS.exists():
        MODEL_WEIGHTS.unlink()
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()

    callbacks = [
        ModelCheckpoint(MODEL_WEIGHTS.as_posix(), save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=25, min_delta=1e-5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, min_delta=1e-5, min_lr=1e-8),
    ]

    start = time.time()
    history = model.fit(
        [X_train_r, m_train],
        y_train_packed,
        validation_data=([X_valid_r, m_valid], y_valid_packed),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )
    print("Training time (s):", time.time() - start)

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df.to_csv("mixed_masked_history.csv", index=False)
    print("Saved training history to mixed_masked_history.csv")

    preds = model.predict([X_test_r, m_test])
    preds *= m_test
    y_test_masked = y_test * m_test

    print("R2:", calculate_r2(y_test_masked, preds))
    print("MAE:", calculate_mae(y_test_masked, preds))
    print("RMSE:", calculate_rmse(y_test_masked, preds))

    model.save(MODEL_PATH.as_posix())
    print("Model saved to", MODEL_PATH)

    final_train_loss, final_val_loss = get_final_losses(history)
    print("Final training loss:", final_train_loss)
    print("Final validation loss:", final_val_loss)
    plot_loss(history)


if __name__ == "__main__":
    main()
