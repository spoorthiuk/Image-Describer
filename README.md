# Image-Describer
This project combines the power of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks using Python to create an image description model. CNNs extract image features, while LSTMs generate descriptive captions. The project uses the Flickr8k dataset, and key steps include data cleaning, feature extraction, tokenizing vocabulary, and creating a data generator. The model architecture involves a feature extractor, sequence processor, and decoder. The provided Python script and GUI allow users to upload images, generate descriptions, and convert them into audio using Google’s Text To Speech API, making images accessible through sound.

## Convolution Neural Network
Convolution Neural Networks (Convents or CNNs) are those neural networks that are able to take an image as the input and learn its various features by applying filters. It is different from a typical Machine Learning model as it does not require any preprocessing and uses principles from linear algebra, specifically matrix multiplication to identify patterns within an image.

They consist of three main layers:

1) Convolution layer
2) Pooling layer
3) Fully-connected layer
As the input image progresses through the different layers, the complexity of the CNN increases and a greater portion of the image is identified.

Convolution Layer

The convolution layer requires the following three things:

Input data
Filter
Feature Map
We will be feeding colored images to generate captions and a colored image is made up of a matrix of pixels in 3D. Therefore, an input image will have a length, breadth and height. A feature detector, also known as a kernel or a filter, is moved across the respective fields of the image checking if the feature is present and this process is known as convolution.

The feature detector is a 2-D array of weights which represents part of the image. When we say that we are applying the filter to a part of the image, it means that a dot product is calculated between the input pixels and the filter. The filter then shifts by a stride and the process is repeated until the kernel has been swept across the entire image. The final dot product output is known as feature map.

After each convolution operation, the CNN applies a Rectified Linear Unit transformation to the feature map thereby introducing nonlinearity to the model.

## Pooling Layer

The pooling layers perform downsampling or dimensionality reduction. In this layer the kernel sweeps across the entire input and applies an aggregation function to the values within the receptive field.

Pooling is mainly of two types:

Max pooling: The pixel with the maximum value is sent to the output array.
Average pooling: The average value of the pixels within the kernel field in sent to the output
This layer reduces complexity, improves efficiency and reduces the risk of overfitting.

## Fully-Connected Layer

This layer performs classification based on the features extracted through the previous layers by usually applying the softmax activation function to classify inputs by producing a probability from 0 to 1.

## Long Short-Term Memory
Long Short-Term Memory (LSTM) Networks are sequential and Recurrent Neural Networks(RNNs) that allow information to persist. RNNs remember the previous information fed to them but one of their shortcomings is that they cannot remember long-term dependencies due to vanishing gradient. This shortcoming is avoided due to the long-term dependency of LSTMs

## LSTM Architecture

The LSTM architecture can be divided into three gates:

Forget Gate: This gate decides if the information coming from the previous timestamp is relevant. It is given by the following equation.


A step function is applied to the result to limit the values to 0 and 1. The transformed is then multiplied with the cell state and the following decisions are made.


Input Gate: This gate decides if we want the current incoming information to be retained by the model for future computation.


New Information: The new information that needs to be passed to the cell state is a function of the hidden state at the previous timestamp and the input at the current timestamp t. It is given by the following equation:

The cell updation will happen with the following equation:


Output gate: This gate completes the information based on the cell state. The equation of the output gate is given by:

The current hidden state is computed using the following equation:


The output is given by the following equation:



## Image Describer Model
In our model, we are using:

CNNs to extract features from the images.
LSTM will use the information from the CNN to generate a description.
Dataset

Flicker8k_Dataset — Dataset folder with 8091 images.
Flickr_8k_Text — Dataset folder containing text files and captions of images.
Project files and folders

Models — This folder stores the trained models.
description.txt — Contains the image captions after pre-processing.
features.p — Pickel object containing the features extracted from the pre-trained Xception CNN model.
tokenizer.p — Contains tokes mapped with index values.
main.py — Python file for generating descriptions for any given image
image-describer.ipynb — Jupyter notebook used in creating the model and training it with the training images.
image-describer.ipynb

### Data Cleaning:
Flickr8k.token.txt file contains the image file names with their respective captions.

Each image contains 5 different captions. We will define a function, img_captions, which will generate a dictionary with the file names as the key and the captions in a list as the values.

In order to clean the captions, we will define another function called data_cleaning to remove punctuation, and words containing numbers and to turn all alphabets into lowercase.

The get_vocabulary function will create a set containing all the words used in the captions. The store_description function will map each caption to the text file name and store it in a text file.

### Extract Features:
We will define a function extract_features that will extract the features from all the images stored in a given directory. We use a pre-trained model called Xception to extract the features from the images. We will resize the given image to [299x299] and normalize the pixel values by first dividing them by 127.5 and subtracting the resulting value with 1.0 to get a value between -1 and 1. After feature extraction, we will store these features in a pickle file called features.p.

### Preparing data for training:
We will define a function called clean_description that will create a dictionary mapping the image file names to their captions after appending <start> and <end> to the captions. load_features will return a dictionary with the mapping of the image names to their respective features extracted from the Xception model.

### Tokenizing vocabulary:
We will define a function called create_tokens that will create word embeddings for all the words used in the captions so that the computer can understand words using Keras Tokenizer() and store these tokens in a pickle file called tokenizer.p

### Data Generator:
We need to now define a function called the data_generator which will create the input and output sequence for our model.

This generator will yield the input and output sequences for a particular image and caption pair as follows


### CNN and RNN model:
The entire model can be divided into three parts:

1) Feature Extractor:
Here we are using the features extracted for each image earlier and converting a vector of size 2048 to 256 nodes using a dense layer. We set the dropout rate to 0.5 in order to prevent overfitting of the model as half (50%) of the neurons in the specified layer will be randomly deactivated (set to zero) at each forward and backward pass.

2) Sequence Processor:
Here we have an embedding layer where it learns a mapping from discrete categorical values (e.g., words) to continuous vector representations followed by a dropout layer to prevent overfitting and an LSTM model.

3) Decoder:
Here we’re gonna merge the outputs of the above two layers by adding them and a dense to arrive at the final prediction

Once we are done designing our model we train it and store the models in the models folder.

main.py

This Python script contains the code to test our models. We have designed a GUI using Tkinter that will enable us to upload images, predict the image description and convert it into audio using Google’s Text To Speech API.

Final Result

![alt text](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*qWgOStNk6pejyiyyVJOR3A.png)


