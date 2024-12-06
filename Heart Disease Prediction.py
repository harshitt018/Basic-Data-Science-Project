import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickles

# Load the dataset
data_path = 'C:/Users/STUDENT/Desktop/Harshit Jaiswal/heart.csv'
data = pd.read_csv(data_path)

# Preprocess the data
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model to a pickle file
model_path = 'C:/Users/STUDENT/Desktop/Harshit Jaiswal/Harshit_Jaiswal_Heart_Disease.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

# Load the model from the pickle file
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Create the main window
root = tk.Tk()
root.title("Heart Disease Prediction")
root.geometry("600x600")
root.configure(bg='#e0f7fa')

# Add a heading
tk.Label(root, text="Heart Disease Predictor", font=('Helvetica', 20, 'bold'), bg='#e0f7fa', fg='#00695c').pack(pady=20)

# Define labels and entries for each feature
features = [
    ("Age", tk.DoubleVar()),
    ("Sex (1=Male, 0=Female)", tk.DoubleVar()),
    ("Chest Pain Type (0-3)", tk.DoubleVar()),
    ("Resting Blood Pressure", tk.DoubleVar()),
    ("Serum Cholesterol", tk.DoubleVar()),
    ("Fasting Blood Sugar (1=True, 0=False)", tk.DoubleVar()),
    ("Resting ECG Results (0-2)", tk.DoubleVar()),
    ("Max Heart Rate Achieved", tk.DoubleVar()),
    ("Exercise Induced Angina (1=Yes, 0=No)", tk.DoubleVar()),
    ("ST Depression", tk.DoubleVar()),
    ("Slope of the Peak Exercise ST Segment (0-2)", tk.DoubleVar()),
    ("Number of Major Vessels (0-3)", tk.DoubleVar()),
    ("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversable Defect)", tk.DoubleVar()),
]

# Create a frame to hold the feature inputs
frame = tk.Frame(root, bg='#e0f7fa')
frame.pack(pady=10)

for i, (label, var) in enumerate(features):
    tk.Label(frame, text=label, font=('Comic Sans MS', 12), bg='#e0f7fa', anchor='w').grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
    tk.Entry(frame, textvariable=var, font=('Arial', 12), bd=2, relief='solid').grid(row=i, column=1, padx=10, pady=5, sticky=tk.W)

# Define the function to predict heart disease
def predict():
    try:
        user_input = [var.get() for _, var in features]
        user_input = np.array(user_input).reshape(1, -1)
        prediction = model.predict(user_input)
        result = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"
        
        # Create a custom message box for the result
        result_window = tk.Toplevel(root)
        result_window.title("Prediction Report")
        result_window.geometry("400x500")
        result_window.configure(bg='#ffffff')

        tk.Label(result_window, text="Prediction Report", font=('Helvetica', 16, 'bold'), fg='#00796b', bg='#ffffff').pack(pady=10)
        
        # Display feature values and prediction result
        result_text = "Prediction Result: " + result + "\n\n"
        result_text += "\n".join([f"{label}: {var.get()}" for label, var in features])
        
        tk.Label(result_window, text=result_text, font=('Times New Roman', 12), bg='#ffffff', justify='left').pack(pady=10, padx=20)
        tk.Button(result_window, text="Close", command=result_window.destroy, font=('Arial', 12), bg='#4caf50', fg='white').pack(pady=20)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create a button to trigger the prediction
predict_button = tk.Button(root, text="Predict", command=predict, font=('Arial', 14), bg='#00796b', fg='white', bd=3, relief='raised')
predict_button.pack(pady=20)

# Run the GUI loop
root.mainloop()
