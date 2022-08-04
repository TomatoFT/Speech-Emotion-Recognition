# Speech Emotion Recognoiton
This is the repository for Final Project of DS102 (Statistical Machine Learning) of University of Information Technology (UIT)
<p>First you need to install Python first</p>
<p>The whole training process is on Google Colab Notebook, just Ctrl F9 to make all things done</p>
<p>Then the colab will download your training models to your local machines, you will need Anaconda to set up Tensorflow more easily.
Do the following steps to run the demo of these algorithms </p>
<h2> Set up the environment to demo </h2>
<p>
The RecordAudio.py will record your voice and save it to output.wav. PredictEmotion.py will predict your Emotion through your audio.</p>
<p> Feel free to clone my code and i will appreciate if you update it to have better performances</p>
# Demo in Command Prompt

<h3>First clone the repository</h3>

```
git clone https://github.com/TomatoFT/Speech-Emotion-Recognoiton

cd Speech-Emotion-Recognoiton

```
<h3>Create Anaconda Virtual Environment And Install Dependencies</h3>

```
conda create --name TrialEnv

conda activate TrialEnv

conda install pip

pip install -r requirements.txt

```
<h3>Set up the demo of your Emotion Prediction of your voice</h3>

```

python RecordAudio.py

python PredictEmotion.py

```

<h3>Exit the Anaconda Virtual Environment </h3>

```
conda deactivate

```

# Demo in Flask Web-app
<h5>Clone the repository and create Anaconda Virtual Environment and install Dependencies same as Command Prompt Demo</h5>

```
python app.py
```
