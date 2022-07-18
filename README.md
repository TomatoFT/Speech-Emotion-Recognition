# Speech-Emotion-Recognoiton
This is the repository for Final Project of DS102 (Statistical Machine Learning) of University of Information Technology (UIT)
<p>First you need to install Python first</p>
<p>The whole training process is on Google Colab Notebook, just Ctrl F9 to make all things done</p>
<p>Then the colab will download your training models to your local machines
Do the following steps to run the demo of these algorithms (except CNN because of my machine is can't setting a GPU for Tensorflow at that moment) </p>
# How to demo
```
git clone https://github.com/TomatoFT/Speech-Emotion-Recognoiton

pip install -r requirements.txt

python RecordAudio.py

python PredictEmotion.py
```

The RecordAudio.py will record your voice and save it to output.wav. PredictEmotion.py will predict your Emotion through your audio.

Feel free to clone my code and i will appreciate if you update it to have better performances
