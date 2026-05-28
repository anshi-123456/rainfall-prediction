# rainfall pprediction
Rainfall Prediction System

This project is a machine learning based Streamlit application that predicts rainfall patterns across different Indian states using synthetic weather data.

The application allows users to input weather parameters such as temperature, humidity, wind speed, month, and state to predict rainfall in millimeters.

Features

Predicts rainfall for 8 Indian states: Maharashtra, Kerala, Tamil Nadu, Rajasthan, Punjab, Karnataka, Uttar Pradesh, and West Bengal

Supports predictions for all 12 months of the year

Uses weather parameters: Temperature (°C) Humidity (%) Wind Speed (km/h) Month State

Machine learning model based on Linear Regression

Requirements

Python 3.8 or higher
pip (Python package manager)
Installation

Clone the repository:

git clone https://github.com/Anujunitian/rainfall_app

cd rainfall-prediction

Create a virtual environment:

python -m venv venv

Activate the virtual environment:

Windows: .\venv\Scripts\activate

macOS / Linux: source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Running the Application

Start the Streamlit app using:

streamlit run rain.py

The application will open in your browser at: http://localhost:8501
