# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:15:16 2025

@author: Abhishek Singh
"""

# Drought Prediction

import pandas as pd
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, SimpleRNN, Dense, Dropout, Flatten
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Input, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
random.seed(11)
np.random.seed(11)
tf.random.set_seed(11)
tf.config.experimental.enable_op_determinism()

# Load dataset
data = pd.read_csv('Rainfall_data.csv')
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data.set_index('Date', inplace=True)
data.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)

# Feature engineering: Add month for seasonality and lagged precipitation
data['Month'] = data.index.month
data['Precipitation_Lag1'] = data['Precipitation'].shift(1)
data['Precipitation_Lag2'] = data['Precipitation'].shift(2)
data['Precipitation_Lag3'] = data['Precipitation'].shift(3)
data = data.fillna(method='bfill')  # Backfill missing lag values

# Define features
features = ['Precipitation', 'Temperature', 'Specific Humidity', 'Relative Humidity', 'Month', 
            'Precipitation_Lag1', 'Precipitation_Lag2', 'Precipitation_Lag3']
data = data[features]

# Handle missing values
data = data.interpolate(method='linear')

# Normalize features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=features, index=data.index).astype(np.float32)

# Create sequences (24-month window)
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predict Precipitation
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

seq_length = 24
X, y = create_sequences(data_scaled.values, seq_length)

# Chronological split: 2000–2016 (training), 2017–2020 (testing)
total_samples = len(X)
train_size = int(0.8 * total_samples)
train_end_date = data.index[train_size + seq_length - 1]
train_X, train_y = X[:train_size], y[:train_size]
test_X, test_y = X[train_size:], y[train_size:]

# Validation split (15% of training, from 2014–2016)
val_size = int(0.15 * train_size)
train_X, val_X = train_X[:-val_size], train_X[-val_size:]
train_y, val_y = train_y[:-val_size], train_y[-val_size:]

# Data augmentation: Add synthetic noise with varying scales
noise = np.random.normal(0, 0.01, train_X.shape).astype(np.float32) * np.random.uniform(0.5, 1.5, train_X.shape).astype(np.float32)
train_X = train_X + noise
train_X = np.clip(train_X, 0, 1).astype(np.float32)

# Define models
def build_rnn():
    model = Sequential([
        SimpleRNN(64, activation='relu', input_shape=(seq_length, len(features))),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return model

def build_cnn():
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, len(features))),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    return model

def build_lstm():
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, len(features))),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return model

def build_cnn_lstm():
    model = Sequential([
        Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(seq_length, len(features)), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling1D(pool_size=2),
        LSTM(75, activation='relu', return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.35),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return model

def build_tcn():
    inputs = Input(shape=(seq_length, len(features)))
    x = inputs
    # Residual block 1
    conv1 = Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu', padding='causal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    conv1_res = Conv1D(filters=64, kernel_size=1, padding='causal')(x)
    x = Add()([conv1, conv1_res])
    # Residual block 2
    conv2 = Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu', padding='causal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    conv2_res = Conv1D(filters=64, kernel_size=1, padding='causal')(x)
    x = Add()([conv2, conv2_res])
    # Residual block 3
    conv3 = Conv1D(filters=64, kernel_size=3, dilation_rate=4, activation='relu', padding='causal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    conv3_res = Conv1D(filters=64, kernel_size=1, padding='causal')(x)
    x = Add()([conv3, conv3_res])
    x = Flatten()(x)
    x = Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Custom loss function for CNN-LSTM, TCN
def custom_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    threshold = 0.25
    weight = tf.where(tf.abs(y_true - threshold) < 0.1, 2.0, 1.0)
    weighted_mse = tf.reduce_mean(weight * tf.square(y_true - y_pred))
    return mse + 0.5 * weighted_mse

# Training and evaluation function
def train_and_evaluate(model, name, test_dates, discriminator=None, discriminator_optimizer=None):
    # Standard model training
    model.compile(optimizer=Adam(learning_rate=0.0002 if name in ['CNN-LSTM', 'TCN'] else 0.0005), 
                  loss=custom_mse if name in ['CNN-LSTM', 'TCN'] else 'mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    history = model.fit(
        train_X, train_y,
        epochs=200,
        batch_size=16,
        validation_data=(val_X, val_y),
        callbacks=[early_stopping, lr_reducer],
        shuffle=False,
        verbose=0
    )
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'{name} Model: Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{name}_loss_plot.png')
    plt.close()

    # Predictions
    train_pred = model.predict(train_X, verbose=0)
    test_pred = model.predict(test_X, verbose=0)
    
    # Inverse transform predictions
    scaler_pred = MinMaxScaler()
    scaler_pred.fit(data[['Precipitation']])
    train_pred = scaler_pred.inverse_transform(train_pred)
    test_pred = scaler_pred.inverse_transform(test_pred)
    train_y_orig = scaler_pred.inverse_transform(train_y.reshape(-1, 1))
    test_y_orig = scaler_pred.inverse_transform(test_y.reshape(-1, 1))
    
    # Metrics
    test_mae = mean_absolute_error(test_y_orig, test_pred)
    test_rmse = np.sqrt(mean_squared_error(test_y_orig, test_pred))
    test_r2 = r2_score(test_y_orig, test_pred)
    
    # Dynamic drought threshold
    combined_y = np.concatenate([train_y_orig, scaler_pred.inverse_transform(val_y.reshape(-1, 1))])
    drought_threshold = np.percentile(combined_y, 25)
    test_drought_true = (test_y_orig < drought_threshold).astype(int)
    test_drought_pred = (test_pred < drought_threshold).astype(int)
    drought_accuracy = np.mean(test_drought_true == test_drought_pred) * 100
    
    # Individual plot
    plt.figure(figsize=(10, 6))
    plt.plot(test_dates, test_y_orig, label='Actual Rainfall', color='blue', linewidth=2)
    plt.plot(test_dates, test_pred, label='Predicted Rainfall', color='red', linestyle='--', linewidth=2)
    plt.axhline(drought_threshold, color='darkgray', linestyle='--', linewidth=1.5, label=f'Drought Threshold: {drought_threshold:.2f} mm')
    plt.text(test_dates[-1], drought_threshold + 5, f'{drought_threshold:.2f} mm', color='darkgray', fontsize=10, ha='right', va='bottom')
    plt.title(f'{name} Model\nActual Vs Predicted Rainfall\nMAE: {test_mae:.2f} mm, RMSE: {test_rmse:.2f} mm, R²: {test_r2*100:.2f}%, Drought Accuracy: {drought_accuracy:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Precipitation (mm)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{name}_plot.png')
    plt.close()
    
    return test_mae, test_rmse, test_r2, drought_accuracy, test_pred, test_y_orig, drought_threshold

# Train and evaluate models
test_dates = data.index[train_size + seq_length:train_size + seq_length + len(test_y)]
models = {
    'RNN': build_rnn(),
    'CNN': build_cnn(),
    'LSTM': build_lstm(),
    'TCN': build_tcn(),
    'CNN-LSTM': build_cnn_lstm(),
}

results = []
all_predictions = {}
test_y_orig_global = None

for name, model in models.items():
    print(f"Training {name}...")
    mae, rmse, r2, drought_acc, test_pred, test_y_orig, drought_threshold = train_and_evaluate(model, name, test_dates)
    results.append([name, mae, rmse, r2 * 100, drought_acc])
    all_predictions[name] = test_pred
    if test_y_orig_global is None:
        test_y_orig_global = test_y_orig  # Store test_y_orig from first model

# Display results
results_df = pd.DataFrame(results, columns=['Model', 'MAE (mm)', 'RMSE (mm)', 'R² (%)', 'Drought Accuracy (%)'])
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Combined plot
plt.figure(figsize=(12, 8))
plt.plot(test_dates, test_y_orig_global, label='Actual Rainfall', color='blue', linewidth=2)
colors = {'RNN': 'orange', 'CNN': 'green', 'LSTM': 'purple', 'CNN-LSTM': 'red', 'TCN': 'magenta'}
for name, pred in all_predictions.items():
    plt.plot(test_dates, pred, label=f'{name} Predicted', color=colors[name], linestyle='--', linewidth=2)
plt.axhline(drought_threshold, color='darkgray', linestyle='--', linewidth=1.5, label=f'Drought Threshold: {drought_threshold:.2f} mm')
plt.text(test_dates[-1], drought_threshold + 5, f'{drought_threshold:.2f} mm', color='darkgray', fontsize=10, ha='right', va='bottom')
plt.title('Model Comparison: Actual vs. Predicted Rainfall')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('combined_plot.png')
plt.close()


'''

Final Results:
    
Model Comparison:
   Model   MAE (mm)  RMSE (mm)    R² (%)  Drought Accuracy (%)
     RNN  83.040382 142.138901 84.015381             73.913043
     CNN  85.943214 149.732819 82.261765             65.217391
    LSTM  99.443443 157.272964 80.430281             78.260870
     TCN 100.646027 188.680725 71.833587             86.956522
CNN-LSTM 103.094070 167.643311 77.764398             84.782609
     
'''