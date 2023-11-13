from flask import Flask, request, jsonify
import numpy as np
import librosa
from tensorflow import keras
import os
from waitress import serve
app = Flask(__name__)

path = r"./audio_classification_model(20,32).h5"
model = keras.models.load_model(path)

def process_audio(audio_data, sample_rate, model):
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    input_spectrogram = np.expand_dims(spectrogram, axis=0)
    input_spectrogram = np.expand_dims(input_spectrogram, axis=-1)
    predicted_prob = model.predict(input_spectrogram)
    predicted_class = 1 if predicted_prob[0][0] >= 0.5 else 0
    return predicted_class 

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
       return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio, sample_rate = librosa.load(audio_file)

    predicted_class = process_audio(audio, sample_rate, model)

    return jsonify({"Predicted class": predicted_class})


if __name__ == '__main__':
    serve(app,host='0.0.0.0',port=8080)
