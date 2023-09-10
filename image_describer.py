import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except Exception as e:
        print("An error occured when opening the image from the specified path \n ERROR: ",e)
    image = image.resize((299,299))
    image = np.array(image)
    #converting 4 channel images to 3
    if image.shape[2] == 4:
        image = image[...,:3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen = max_length)
        pred = model.predict([photo,sequence], verbose = 0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


file_name = 'Flicker8k_Dataset/2356574282_5078f08b58.jpg'
max_length = 34
tokenizer = load(open("tokenizer.p",'rb'))
model = load_model('models/model_1.h5')
xception_model = Xception(include_top = False, pooling = 'avg')
photo = extract_features(file_name, xception_model)
img = Image.open(file_name)
description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)