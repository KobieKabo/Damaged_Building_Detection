# Damaged Building Prediction with Nueral Networks

This repository works with Neural Network Architectures for machine learning models to classify sattelite images of Texas buildings following hurricane Harvey. The models split the data into a binary classification of 'damaged' and 'no damage' buildings with the assistance of Tensorflow neural networks. In this repository we explored three different NN architectures: Artifical Neural Network (ANN), and Two different LeNet5 CNN models. We then deployed the best performing model with a inference server hosted on a flask server, that can be easily deployed with the assistance of a docker image. Flask API calls & examples are shown further in this README.


# TensorFlow Model Serving API

## Overview

This API serves as an interface for classifying images using pre-trained TensorFlow models. It allows users to classify images as "damaged" or "not damaged", change the underlying model, and retrieve information about the current model.

## Use

### API Endpoints

| Endpoint            | Method | Description                                                                                         |
|---------------------|--------|-----------------------------------------------------------------------------------------------------|
|       |     |  |
|     |     | |

### Running with Docker

The  Docker image for the inference server is ``.

Run:
```
$ docker pull {insert image}
$ docker run -p 5000:5000 {insert image}  
```

### Making Requests to the Inference Server

#### Requesting Model Information
To retrieve information about the currently loaded model, you can make a GET request to the /model/info endpoint:

```

{
  
}
```
This request will return JSON data containing details such as the model's version, name, description, and parameter counts.

#### Classifying an Image
To classify an image using the inference server, send a POST request to the /model/predict endpoint with the image you want to classify. Here's an example using curl:

```

```
for example:
```

```

### Architectures:

We experimented with three different neural network architectures:

- **ANN (Artificial Neural Network)**: A dense, fully-connected network with varying layers and perceptrons.

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 49152)             0         
                                                                 
 dense (Dense)               (None, 256)               12583168  
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 64)                8256      
                                                                 
 dense_3 (Dense)             (None, 32)                2080      
                                                                 
 dense_4 (Dense)             (None, 16)                528       
                                                                 
 dense_5 (Dense)             (None, 8)                 136       
                                                                 
 dense_6 (Dense)             (None, 2)                 18        
                                                                 
=================================================================
Total params: 12627082 (48.17 MB)
Trainable params: 12627082 (48.17 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

- **LeNet-5 CNN**: A convolutional neural network based on the classical Lenet-5 architecture, adjusted for our image dimensions.

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 126, 126, 6)       168       
                                                                 
 average_pooling2d (Average  (None, 63, 63, 6)         0         
 Pooling2D)                                                      
                                                                 
 conv2d_1 (Conv2D)           (None, 61, 61, 16)        880       
                                                                 
 average_pooling2d_1 (Avera  (None, 30, 30, 16)        0         
 gePooling2D)                                                    
                                                                 
 flatten (Flatten)           (None, 14400)             0         
                                                                 
 dense (Dense)               (None, 120)               1728120   
                                                                 
 dense_1 (Dense)             (None, 84)                10164     
                                                                 
 dense_2 (Dense)             (None, 2)                 170       
                                                                 
=================================================================
Total params: 1739502 (6.64 MB)
Trainable params: 1739502 (6.64 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

- **Alternate-LeNet-5 CNN**: Inspired by a variant discussed in [this research paper](https://arxiv.org/pdf/1807.01688.pdf), though customized for our specific dataset.

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_2 (Conv2D)           (None, 126, 126, 6)       168       
                                                                 
 max_pooling2d (MaxPooling2  (None, 63, 63, 6)         0         
 D)                                                              
                                                                 
 conv2d_3 (Conv2D)           (None, 61, 61, 32)        1760      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 30, 30, 32)        0         
 g2D)                                                            
                                                                 
 conv2d_4 (Conv2D)           (None, 28, 28, 64)        18496     
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 14, 14, 64)        0         
 g2D)                                                            
                                                                 
 conv2d_5 (Conv2D)           (None, 12, 12, 128)       73856     
                                                                 
 max_pooling2d_3 (MaxPoolin  (None, 6, 6, 128)         0         
 g2D)                                                            
                                                                 
 conv2d_6 (Conv2D)           (None, 4, 4, 128)         147584    
                                                                 
 max_pooling2d_4 (MaxPoolin  (None, 2, 2, 128)         0         
 g2D)                                                            
                                                                 
 flatten_1 (Flatten)         (None, 512)               0         
                                                                 
 dropout (Dropout)           (None, 512)               0         
                                                                 
 dense_3 (Dense)             (None, 120)               61560     
                                                                 
 dense_4 (Dense)             (None, 84)                10164     
                                                                 
 dense_5 (Dense)             (None, 2)                 170       
                                                                 
=================================================================
Total params: 313758 (1.20 MB)
Trainable params: 313758 (1.20 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
