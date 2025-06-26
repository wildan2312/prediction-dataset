import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Tambahkan semua class model yang digunakan
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import Gaussian

# Load semua model
with open('model_knn.pkl', 'rb') as f:
    model_knn = pickle.load(f)
with open('model_dt.pkl', 'rb') as f:
    model_dt = pickle.load(f)
with open('model_nb.pkl', 'rb') as f:
    model_nb = pickle.load(f)

# Load dataset untuk mengambil fitur
df = pd.read_csv("ilpd.csv", header=None)
df.columns = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'SGPT', 'SGOT',
              'TP', 'ALB', 'A/G Ratio', 'Dataset']

# Preprocessing seperti sebelumnya
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Dataset'] = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)
df.fillna(df.mean(), inplace=True)

X = df.drop('Dataset', axis=1)
y = df['Dataset']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Form input
st.title("Prediksi Penyakit Hati")

# Buat form input user
age = st.slider('Umur', 1, 100, 45)
gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
tb = st.number_input('Total Bilirubin', 0.0, 10.0, 1.0)
db = st.number_input('Direct Bilirubin', 0.0, 5.0, 0.5)
alkphos = st.number_input('Alkaline Phosphotase', 0, 2000, 200)
sgpt = st.number_input('SGPT', 0, 2000, 50)
sgot = st.number_input('SGOT', 0, 2000, 50)
tp = st.number_input('Total Protein', 0.0, 10.0, 6.5)
alb = st.number_input('Albumin', 0.0, 6.0, 3.0)
ag = st.number_input('A/G Ratio', 0.0, 3.0, 1.0)

# Pilih model
model_choice = st.selectbox("Pilih Model", ["KNN", "Decision Tree", "Naive Bayes"])

# Prediksi
if st.button("Prediksi"):
    gender_val = 1 if gender == "Male" else 0
    input_data = np.array([[age, gender_val, tb, db, alkphos, sgpt, sgot, tp, alb, ag]])
    input_scaled = scaler.transform(input_data)

    if model_choice == "KNN":
        pred = model_knn.predict(input_scaled)[0]
    elif model_choice == "Decision Tree":
        pred = model_dt.predict(input_scaled)[0]
    else:
        pred = model_nb.predict(input_scaled)[0]

    if pred == 1:
        st.error("Hasil: Pasien Terindikasi Penyakit Hati")
    else:
        st.success("Hasil: Pasien Sehat")
