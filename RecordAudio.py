import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 4  # Duration of recording
print("-----------------------START RECORDING-------------------------")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
print("-----------------------STOP RECORDING-------------------------")
write('output.wav', fs, myrecording)  # Save as WAV file