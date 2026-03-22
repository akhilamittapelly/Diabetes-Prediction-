import joblib
import numpy as np
from django.shortcuts import render
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Load the pre-trained Random Forest model and the pre-fitted transformers (scaler and power transformer)
rf_model = joblib.load('model/rf_model.pkl')
scaler = joblib.load('model/scaler.pkl')  # Make sure you saved the fitted scaler during model training
pt = joblib.load('model/pt.pkl')  # Similarly, save the fitted PowerTransformer during training

def home(request):
    prediction = None 

    if request.method == 'POST':
        # Get the features from the form and convert them to floats
        Pregnancies = float(request.POST.get('Pregnancies'))
        Glucose = float(request.POST.get('Glucose'))
        BloodPressure = float(request.POST.get('BloodPressure'))
        SkinThickness = float(request.POST.get('SkinThickness'))
        Insulin = float(request.POST.get('Insulin'))
        BMI = float(request.POST.get('BMI'))
        DiabetesPedigreeFunction = float(request.POST.get('DiabetesPedigreeFunction'))
        Age = float(request.POST.get('Age'))

        # Create the feature vector (input array) for prediction
        features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, 
                             Insulin, BMI, DiabetesPedigreeFunction, Age]])

        # Apply PowerTransformer and StandardScaler
        transformed_data = pt.transform(features)  # Apply power transformation
        features_scaled = scaler.transform(transformed_data)  # Standardize the input features

        # Make prediction using the loaded Random Forest model
        rf_prediction = rf_model.predict(features_scaled)
        print("\nRandom Forest Prediction (0 = No, 1 = Yes):", rf_prediction[0])
        if  rf_prediction[0] == 0:
            result = 'No Diabetes'
        else:
            result = 'Diabetes'
        # Render the result page with the prediction result
        return render(request, 'result.html', {'prediction_text': result})

    return render(request, 'home.html')
