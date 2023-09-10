import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import os
from gtts import gTTS

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

def predict_model(filename):
    max_length = 34
    tokenizer = load(open("tokenizer.p",'rb'))
    model = load_model('models/model_9.h5')
    xception_model = Xception(include_top = False, pooling = 'avg')
    photo = extract_features(filename, xception_model)
    description = generate_desc(model, tokenizer, photo, max_length)
    print("\nGenerating description...\n")
    print('Predicted description : {} \n'.format(description))
    print('Google GTTS audio...')
    audio_description = ' '.join(description.split(' ')[1:-1])
    text_to_speech(audio_description)

def text_to_speech(text):
    speech = gTTS(text)
    speech_file = 'speech.mp3'
    speech.save(speech_file)
    os.system('afplay ' + speech_file)

window = tk.Tk()
window.geometry("500x500")  # Size of the window 
window.title('Image Describer')
my_font1=('times', 18, 'bold')
l1 = tk.Label(window,text='Image Describer',width=30,font=my_font1,bg='#fff', fg='#f00')  
l1.grid(row=1,column=1)
b1 = tk.Button(window, text='Upload Image', 
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1) 

def upload_file():
    global img
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    print(filename)
    img = ImageTk.PhotoImage(file=filename)
    b2 =tk.Button(window,image=img) # using Button 
    b2.grid(row=3,column=1)
    predict_model(filename)

window.mainloop()