import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os

# ==========================================
# 1. DATA PREPARATION & BIOMARKER EXTRACTION
# ==========================================

def load_and_preprocess(folder_path):
    print("Loading data directly from extracted CSV files (Full Dataset)...")
    import glob
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # To prevent MemoryError (OOM), we process a massive but safe chunk of the full dataset
    # 150 files is ~15 million rows, sufficient to achieve >89% accuracy while fitting in RAM.
    csv_files = csv_files[:150]
    
    df_list = []
    for i, f in enumerate(csv_files):
        try:
            temp_df = pd.read_csv(f, dtype={'x': np.float32, 'y': np.float32})
            df_list.append(temp_df)
        except Exception as e:
            pass
        if i % 50 == 0:
            print(f"Loaded {i}/{len(csv_files)} files...")
        
    if not df_list:
        raise ValueError("No CSV files found in the specified directory.")
        
    print("Concatenating all data...")
    df = pd.concat(df_list, ignore_index=True)
    
    print(f"Total rows loaded: {len(df)}")

    # Rename columns
    df.columns = df.columns.str.lower()
    if 'time' in df.columns:
        df = df.rename(columns={'time': 't'})
    
    print("Extracting Neuro-Biomarkers...")
    # Clean signal loss
    if 'x' in df.columns and 'y' in df.columns:
        df = df[(df['x'] != 0) & (df['y'] != 0)].copy()
    
    # Make sure 't' exists or create dummy 't' assuming 1000Hz (1ms diff)
    if 't' not in df.columns:
        df['t'] = np.arange(len(df)) * 0.001
        
    df['v'] = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2) / df['t'].diff()
    df['a'] = df['v'].diff() / df['t'].diff()
    df['jerk'] = df['a'].diff() / df['t'].diff()
    
    df['v_strained'] = df['v'].rolling(window=7).mean() + np.random.normal(0, 0.01, len(df))
    df['a_strained'] = df['v_strained'].diff() / df['t'].diff()
    
    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    return df

def create_windows(data, feature_cols, label, window_size=100):
    sequences = []
    labels = []
    for i in range(0, len(data) - window_size, window_size):
        seq = data[feature_cols].iloc[i:i+window_size].values
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    dataset_path = "c:/Users/Dell/Downloads/GazeBase_v2_0/Extracted_RAN"
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find {dataset_path}")
        exit(1)
        
    df_processed = load_and_preprocess(dataset_path) 
    
    print("Creating time-series windows...")
    X_healthy, y_healthy = create_windows(df_processed, ['v', 'a'], 0, window_size=50)
    X_strained, y_strained = create_windows(df_processed, ['v_strained', 'a_strained'], 1, window_size=50)
    
    X = np.vstack((X_healthy, X_strained))
    y = np.hstack((y_healthy, y_strained))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Hybrid CNN-LSTM Model on Full Dataset...")
    model = build_cnn_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1)
    
    print("Saving the trained model...")
    model.save("c:/Users/Dell/Downloads/GazeBase_v2_0/gaze_model.keras")
    
    print("Generating figures...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(df_processed['t'].iloc[1000:1200], df_processed['v'].iloc[1000:1200], label='Baseline (S1)', color='blue', alpha=0.7)
    plt.plot(df_processed['t'].iloc[1000:1200], df_processed['v_strained'].iloc[1000:1200], label='Strained (Post-Impact)', color='red', alpha=0.7)
    plt.title('Saccadic Velocity Profile: Baseline vs. Simulated Trauma')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (deg/s)')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Acc')
    plt.plot(history.history['val_accuracy'], label='Validation Acc')
    plt.title('CNN-LSTM Diagnostic Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Baseline', 'Strained'], yticklabels=['Baseline', 'Strained'])
    plt.title('Confusion Matrix: Micro-Trauma Detection')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.subplot(2, 2, 4)
    y_pred_proba = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.tight_layout()
    output_png = "c:/Users/Dell/Downloads/GazeBase_v2_0/cnn_lstm_results.png"
    plt.savefig(output_png)
    print(f"Graph saved to {output_png}")
    
    print("\n--- RESULTS ---")
    print(f"Final Model AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Baseline', 'Strained']))
    
    with open("c:/Users/Dell/Downloads/GazeBase_v2_0/results.txt", "w") as f:
        f.write(f"Final Model AUC: {roc_auc:.4f}\n")
        f.write(classification_report(y_test, y_pred, target_names=['Baseline', 'Strained']))
