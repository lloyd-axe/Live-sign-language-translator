# Live Sign Language Translator
Sign language is a form of manual communication that is commonly used by deaf people. This project is a beginner's attempt in demonstrating the use of deep learning for real world problems. 
<br/>
![demo](https://user-images.githubusercontent.com/67902015/152925333-8fdfeab3-2218-4196-b091-82208350bdf6.gif)
<br/>

This implementation is technically a live **hand-shoulder gesture recognizer** because it can be trained to recognize any gesture. But, its main purpose is for translating sign language.


## How it works
Using the [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html) model, **138** keypoints are extracted from the person's **hands** and **shoulders**. This data is then used to train a basic model that can identify multiple gestures. 

## How to Use
_For the complete implementation, you can check the [Test.ipynb](https://github.com/lloyd-axe/Live-sign-language-translator/blob/main/Test.ipynb) notebook._

First, you'll have to install all the required packages
```
pip install -r requirements.txt
```

Then, make sure you define both the model's and word list's paths when initializing the **ASLTranslator** object.
To start live capture, use the **start_capture** method.
```python
aslt = ASLTranslator(MODEL_PATH, WORD_LIST_PATH)
aslt.start_capture()
```

### Data Collection, Model Creation and Training using ASLTranslator
You can collect fresh data by using the **collect_word_data** method.
```python
words = ['hello', 'yes', 'no']
data_path = 'test_data'
for word in words:
    aslt.collect_word_data(word, data_path, samp_count = 5, frame_count = 10)
```

For model creation and training, you can use the **create_model** and **train_model** methods as shown below:
```python
with open(os.path.join(data_path, 'word_list.txt')) as file:
    words = [line.rstrip() for line in file]
model = aslt.create_model(len(words), interval = 10)
x_train, x_test, x_val, y_train, y_test, y_val = aslt.split_data(sequences, labels)
train_data = (x_train, y_train)
val_data = (x_val, y_val)
aslt.train_model(model, train_data, val_data, plot = True, epochs = 100, model_name='test_model.h5')
```

### Website Installation
1. Open [Website](https://github.com/lloyd-axe/Live-sign-language-translator/tree/main/Website) folder.
2. (Optional) Create a virtual environment.
3. Install packages listed in its [requirements.txt](https://github.com/lloyd-axe/Live-sign-language-translator/blob/main/Website/requirements.txt).
```
pip install -r requirements.txt
```
4. Attach **model.h5** and **word_list.txt** files to **Website/project_sigua/static/** folder
5. Run server.

```
python manage.py runserver
```

## Possible Improvements
* Use other techniques like convolutional neural network
* Larger training datasets
