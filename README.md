# phenumoia-detction01



required python lib 
streamlit
keras
PIL
tensorflow 
base64
numpy
pandas
pillow


# Detection of Pneumonia using ML & DL in Python

#Introduction

7% of the population or 450 million people worldwide are suffering from Pneumonia which results in
the death of 4 million people every year . Pneumonia is an acute lower respiratory disease with
traces of infection usually because of some micro-organism, bacteria or virus. World health
organization (WHO) states that the pneumonia is the leading reason for the child dearth in the world. It
had killed approximately 1.2 million children under the age of five. And south Asia is leading the
table. South Asian countries like Indian Pakistan, Bangladesh, Sri Lanka, Indonesia etc. have the most
no. children having pneumonia disease. Not only is this death rate of pneumonia greater than death rate
of other dangerous disease like AIDS, malaria and tuberculosis combined . Among children younger
than five years, pneumonia is considered as one of the deadliest diseases . Only India has
encountered 158,176 deaths in the year 2016, and still India continue to show the significant no.in the
infant deaths because of pneumonia. According to WHO report published on World would be
consumed by this communicable disease. In today’s time the most effective Pneumonia detection
method accepted by the doctors is radiograph. Although, some studies tell us that discrepancies are
usual in the interpretation of the chest radiographs via radiologist. This drawback of human based
observation has provided enough motivation to technology making interference in this field where
machine will classify between a normal X-ray and abnormal Chest X-ray. This provides accuracy and speed to the diagnosis process.  Our problem for this project is the “binary Classification” of the
input chest X-ray images and then classifies them between the two classes of pneumonia or non-
pneumonia. This project emphasizes upon the extensive use of neural networks in detection of
pneumonia in chest X-rays.



<img width="797" alt="Screenshot 2024-03-16 at 5 23 20 AM" src="https://github.com/Nirbhaypatil24/phenumoia-detction01/assets/160317380/faaaf5b5-fe78-4b4c-92a6-980c5f629a28">



#Literature Review

After analyzing and reading various datasets available on various platforms and websites, pneumonia
dataset was found to be best fit for performing on and making a model to detect it using image dataset
of chest X-rays of patients . World health organization (WHO) states that the pneumonia is the
leading reason for the child dearth in the world. It had killed approximately 1.2 million children under
the age of five. And south Asia is leading the table. South Asian countries like Indian Pakistan,
Bangladesh, Sri Lanka, Indonesia etc. have the most no. children having pneumonia disease. Not only
is this death rate of pneumonia greater than death rate of other dangerous disease like AIDS, malaria
and tuberculosis combined . Pneumonia is one of the gravest illnesses among children younger than
5 years of age . This was motivation enough to work on this dataset and produce a model with
accuracy good enough to successfully detect pneumonia by reading chest X-rays While examining for
pneumonia in the patient’s X-ray, the radiologist looks in it for spots, specifically white ones, within
the lungs termed as “infiltrates” which are helpful in identifying the infection.
Pneumonia chest x-ray can be observed in TB, severe case of bronchitis as well. Complete Blood
Count (CBC), Chest Computed Tomography (CT) and sputum test etc. are further conducted to reach a
conclusion about the infection. Therefore, in this attempt to solve the problem we have only tried to
detect whether a chest x-ray conclude that a person is ill with pneumonia or normal patients and do so
by searching for any cloudy pattern in the X-ray. Conclusive detection will therefore, depend on
pathological tests only . Today many diseases are detected using Artificial Intelligence based
solution. Some of these diseases are breast cancer, brain tumor etc. based solutions . This Artificial
Intelligence based detection by Convolutional Neural Network have played a great part and shown
great promises in classification and this AI based classification in trusted by doctors all over the world.
When we talk about low-cost imaging methods and easy use, Deep and machine learning methods are
gaining popularity when it comes to examining chest X-rays. Also, the fact that there is ample of data
available for training of various machine learning models. Among all the papers studied by us, the
highest accuracy was obtained by  as 98%.
Thus, we chose CNN for operating on our dataset and took the help of this deep learning approach to
obtain accuracy better than other models using other deep learning approaches.

#Theory
#CNN:
Convolutional Neural Network or CNN, due to its better than ever efficiency in Image
classification, is becoming quite popular now-a- days. Features such as spatial and temporal in an
image are easily extracted using CNN. Each layer has some specific weight as CNN use weightsharing technique as it reduces the computation efforts [6-7] Architecture wise, Convolutional Neural
Network are simply forward feed artificial neural Network (ANN) with only two limitations. To
reduce the total number of parameters of the model, firstly the weights should be shared and secondly
neurons which are in the same layers are connected to local paths only. There are three basic building
blocks in CNN to preserve the spatial structure:
 A convolution layer.
 A max-pooling layer to reduce the Dimension of the map
 A fully connected layer to classify the between the kinds [8]. Below in figure 2, the
architectural explanation of CNN is given.

<img width="797" alt="Screenshot 2024-03-16 at 5 28 53 AM" src="https://github.com/Nirbhaypatil24/phenumoia-detction01/assets/160317380/81a3200c-3e73-4667-a98c-9f473421f97d">


#Normalization:
Normalization has been a full of life research field of deep learning normalization is
used to reduce the training time by a lot of factors, allow us to show a number of the advantages of
using Normalization.
 Sometimes some features have a very high value as compared to the feature surrounding it, so
in this process we normalize each and every feature is maintained. By doing so, we’re making
our network unbiased.
 Normalization decreases the internal covariate shift. Covariate shift is the change within the
distribution of activation network .and because of the alteration in the network parameters
during training of the model. To boost the training, we need to scale back the Internal
Covariate Shift
 As stated in point no. 1 normalization maintains the features but the sharp feature is lost
because of normalization
 Normalization paces up the Optimization as normalization do not allow weights to explode
and it also restrict them to a specific range
 Normalization also helps CNN in regularization which is one unintended benefit of it (only
slightly, not significantly though) [10].
The normalization technique used in our model was Batch Normalization.

#Pre-processing: 
In this we have resized the images for performing efficient operation on them. The
images are originally 1000 pixels per dimension and they were resized to a compatible size for better
computation. We also created a function in which pneumonia file in training dataset is given label = 0
and normal label = 1, else = 2. Two arrays, ‘X’ and ‘Y’ are created where ‘X’ stores pre-processed
(resized image) data and ‘Y’ stores the label which is again stored back in the file we use the function
‘resize’ of OpenCV, and uploading the images in new h5py file whose main function is to store the
data in binary format, means images of chest X-ray are converted to array of numbers as in figure .

<img width="428" alt="Screenshot 2024-03-16 at 5 31 36 AM" src="https://github.com/Nirbhaypatil24/phenumoia-detction01/assets/160317380/a23d425d-2e7b-4ac5-9591-87bc33553256">

The above image of the chest X-ray above is changed in an array like given figure


<img width="460" alt="Screenshot 2024-03-16 at 5 30 56 AM" src="https://github.com/Nirbhaypatil24/phenumoia-detction01/assets/160317380/ad53d43b-c770-4057-8acf-372d9435945c">





