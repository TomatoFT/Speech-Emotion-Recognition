from flask import Flask, render_template, request
import sounddevice as sd
from scipy.io.wavfile import write
import sounddevice as sd
import soundfile as sf
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
import joblib
import librosa
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import sys
import warnings
app = Flask(__name__, template_folder='templates')

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/recording',  methods=['GET', 'POST'])
def recording():
    fs = 44100  # Sample rate
    seconds = 4  # Duration of recording
    if request.method == "POST":
        print("-----------------------START RECORDING-------------------------")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  
        print("-----------------------STOP RECORDING-------------------------")
        write('static/output.wav', fs, myrecording)  
    return render_template('recording.html')

@app.route('/predicting', methods=['GET', 'POST'])
def predicting():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    data, sample_rate = librosa.load('static/output.wav')
    def Decoding(list):
        if list[0] == 0:
            return "Angry"
        elif list[0] == 1:
            return "Fear"
        elif list[0] == 2:
            return "Surprise"
        elif list[0] == 3:
            return "Disgust"
        elif list[0] == 4:
            return "Calm"
        elif list[0] == 5:
            return "Happy"
        elif list[0] == 6:
            return "Sad"
        else:
            return "Neutral"
    def noise(data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(data, rate=0.8):
        return librosa.effects.time_stretch(data, rate)

    def shift(data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    def extract_features(data):
        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr)) # stacking horizontally

        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft)) # stacking horizontally

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc)) # stacking horizontally

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms)) # stacking horizontally

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel)) # stacking horizontally
        
        return result

    def get_features(path):
        # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
        data, sample_rate = librosa.load(path, duration=4, offset=0.6)
        
        # without augmentation
        res1 = extract_features(data)
        result = np.array(res1)
        
        # # data with noise
        # noise_data = noise(data)
        # res2 = extract_features(noise_data)
        # result = np.array((result, res2) # stacking vertically
        
        # # # data with stretching and pitching
        # new_data = stretch(data)
        # data_stretch_pitch = pitch(new_data, sample_rate)
        # res3 = extract_features(data_stretch_pitch)
        # result = np.array(res3) # stacking vertically
        
        return result

    # # Extract data and sampling rate from file
    # data, fs = sf.read(filename, dtype='float32')  
    # sd.play(data, fs)
    # status = sd.wait()  # Wait until file is done playing
    CNN = tf.keras.models.load_model('Model/content/CNN')
    LR = joblib.load("Model/LogisticRegresion.pkl")
    SVM = joblib.load("Model/SVM.pkl")
    Knn = joblib.load("Model/Knn.pkl")
    test_case = 'static/output.wav'
    featuring = get_features(test_case)
    X3 = [featuring]
    X4 = np.expand_dims(X3, axis=2)
    nl = []
    # scaler = StandardScaler()
    # X3 = scaler.fit_transform(X3)
    # X3 = X3.reshape(1, -1)
    pred_test_1 = LR.predict(X3)
    pred_test_2 = SVM.predict(X3)
    pred_test_3 = Knn.predict(X3)
    pred_test_4 = CNN.predict(X4)
    pred_test_4 = pred_test_4.flatten()
    print('pred_test_3',pred_test_3)
    i=0
    for pred in pred_test_4:
        nl.append(pred)
    index_max = max(nl)
    # print(index_max)
    # print(nl)
    for pred in nl:
        if pred == index_max:
            print(i, type(i))
        else:
            i+=1
    print(Decoding(np.ndarray(i)))
    # pred_test_4 = CNN.predict(X3)
    # encoder2.fit_transform(np.array(Y1).reshape(-1, 1))
    # y_pred2 = encoder2.inverse_transform(pred_test.reshape(-1, 1))
    # print(f"Logistic Regression: {Decoding(pred_test_1)},SVM: {Decoding(pred_test_2)}, K-Nearest Neighbour: {Decoding(pred_test_3)}")
    df = pd.DataFrame({"Logistic Regression":Decoding(pred_test_1),
                        "SVM": Decoding(pred_test_2),
                        "Knn": Decoding(pred_test_3),
                        'CNN': Decoding(np.ndarray(i))
                        }, index=["Prediction"])
    print(df)
    predictions = {"LogisticRegression":Decoding(pred_test_1),
                        "SVM": Decoding(pred_test_2),
                        "Knn": Decoding(pred_test_3),
                        'CNN': Decoding(np.ndarray(i))
                        }
    return render_template('predict.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)