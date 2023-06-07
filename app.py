import pandas as pd
from flask import Flask, render_template, request
import scipy.io as sio
from scipy.stats import kurtosis
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

app = Flask(__name__)
model = pickle.load(open("model_tuning.pkl", "rb"))
folder = app.config["UPLOAD_FOLDER"] = "uploads"

Data_type = float

#Membuat fungsi untuk menghitung fitur time domain
def calculate_time_domain_features(data_vibrasi):
    Maximum_Absolut = np.max(np.abs(data_vibrasi))
    Faktor_Puncak = Maximum_Absolut / (np.sqrt(np.mean(np.square(data_vibrasi))))
    Faktor_Shape = (np.sqrt(np.mean(np.square(data_vibrasi)))) /(np.abs(data_vibrasi).mean())
    Impulse = Maximum_Absolut / (np.abs(data_vibrasi).mean()) #

    return Maximum_Absolut, Faktor_Puncak, Faktor_Shape, Impulse

#Membuat fungsi untuk normalisasi data time domain
def normalisasi_data(fitur):
    min_values = {
        'Maximum_Absolut': 8.971681,
        'Faktor_Puncak': 4.693771,
        'Faktor_Shape': 1.253648,
        'Impulse': 5.891499
    }

    max_values = {
        'Maximum_Absolut': 22.969166,
        'Faktor_Puncak': 7.992036,
        'Faktor_Shape': 1.339602,
        'Impulse': 10.284297
    }

    normalized_data = []
    for i, feature in enumerate(fitur):
        feature_name = list(min_values.keys())[i]
        min_val = min_values[feature_name]
        max_val = max_values[feature_name]
        normalized_value = (feature - min_val) / (max_val - min_val)
        normalized_data.append(normalized_value)
    return normalized_data

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    #Load matlab file
    file = request.files["file"]
    file_path = os.path.join(folder, file.filename)
    file.save(file_path)
    mat_data = sio.loadmat(file_path)

    #Ambil data vibrasi dari file matlab
    vibrasi = mat_data["vibration"]

    #Menghitung fitur time domain dari data vibrasi
    fitur_time_domain = calculate_time_domain_features(vibrasi)

    #Normalisasi fitur time domain
    normalisasi_fitur_time_domain = normalisasi_data(fitur_time_domain)
    normalisasi_fitur_time_domain = [round(item, 6) for item in normalisasi_fitur_time_domain]

    #Ubah list ke dalam bentuk array
    X = np.array(normalisasi_fitur_time_domain).reshape(1, -1)

    #Prediksi
    prediction = model.predict(X)

    fitur_names = ['Maximum_Absolut', 'Faktor_Puncak', 'Faktor_Shape', 'Impulse']
    fitur_values = normalisasi_fitur_time_domain
    fitur_dict = dict(zip(fitur_names, fitur_values))


    return render_template("index.html", prediction_text = "{} Day left".format(int(prediction)), fitur_dict=fitur_dict)