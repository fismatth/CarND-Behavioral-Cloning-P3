**Write-up Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

**1. Submitted Files**

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* autonomous_driving_record.mp4 record of driving in autonomous mode

**2. Driving in autonomous mode**
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

**3. Description of model**

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

**3.1. Model architecture**

I started with the LeNet architecture and adapted the parameters and added some layers to make the network more powerful. My final model consists of the following layers:

* Normalization (using Keras lamda layer)
* 2 Convolutional layers with RELU activation (introduce non-linearity) and max pooling with filter size 3x3
* 2 Convolutional layers with RELU activation and max pooling with filter size 2x2
* Flattening
* 4 fully connected layers with dropout layers (to reduce overfitting) in-between

**3.2. Extending and tuning the model**

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 
Where this was not the case, I tried to extend and tune the model and/or generate more training data (see next two sections).
With the final model, the car was able to stay on track for several laps (see recorded video autonomous_driving_record.mp4).

**3.3. Model parameter tuning**

The model used an adam optimizer, so the learning rate was not tuned manually. 
Other parameters of the model that have been tuned are the filter sizes of the convolutional layers and the output size of the (first three) fully connected layers. 
I did choose relatively small filter sizes for the convolutional layers and relatively large output sizes of the fully connected layers to make the network more powerful.

**3.4. Appropriate training data**

I used the provided training data and concentrated on generating additional data from this data set using the following techniques:

* Use the left camera image for positive steering angles and the right image for negative steering angles to cover the extreme cases (recovering when we are on the outside of the road). The angle offset (1.0) has been choosen relatively high which seems to enable the model to even recover from difficult cases. On the other hand, this sometimes leads to sine-curve-like driving along the center line of the road.
* Flip images to avoid (in this case) left turn bias
* Blurr images with a gaussian kernel of size 5x5, especially to cover cases where the borders of the road are not that clear (e.g. right side of the curve after the bridge)
