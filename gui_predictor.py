import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# GUI Functionality
def predict():
    try:
        # Collect user inputs
        open_price = float(entry_open.get())
        high_price = float(entry_high.get())
        low_price = float(entry_low.get())
        close_price = float(entry_close.get())
        volume = float(entry_volume.get())
        ma_10 = float(entry_ma10.get())
        ma_50 = float(entry_ma50.get())

        # Prepare the input data
        input_data = pd.DataFrame([{
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume,
            'MA_10': ma_10,
            'MA_50': ma_50
        }])
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        prediction_prob = model.predict_proba(input_data_scaled)[0]

        # Display results
        result = "Buy" if prediction == 1 else "Sell"
        confidence = max(prediction_prob) * 100
        result_label.config(text=f"Prediction: {result}\nConfidence: {confidence:.2f}%")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# GUI Setup
root = tk.Tk()
root.title("Trading Model Predictor")
root.geometry("400x400")

# Title Label
title_label = ttk.Label(root, text="Trading Model Predictor", font=("Arial", 16))
title_label.pack(pady=10)

# Input Fields
frame_inputs = ttk.Frame(root)
frame_inputs.pack(pady=10)

labels = ["Open Price", "High Price", "Low Price", "Close Price", "Volume", "10-Day MA", "50-Day MA"]
entries = []

for label_text in labels:
    label = ttk.Label(frame_inputs, text=label_text)
    label.grid(sticky="W", padx=5, pady=2)
    entry = ttk.Entry(frame_inputs)
    entry.grid(row=labels.index(label_text), column=1, padx=5, pady=2)
    entries.append(entry)

entry_open, entry_high, entry_low, entry_close, entry_volume, entry_ma10, entry_ma50 = entries

# Predict Button
predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=10)

# Result Label
result_label = ttk.Label(root, text="Prediction: --\nConfidence: --", font=("Arial", 14), foreground="blue")
result_label.pack(pady=20)

# Run the app
root.mainloop()
