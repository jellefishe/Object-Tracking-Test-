# Object-Tracking-Test-
Robust Object Tracker Given a data set

Libs used: numPy, OpenCV, Pillow/PIL

From the Umich EECS442 Computer Vision course I have learned a variety of ways to process images and videos into various types of data. I used the Kalman filter technique to identify the object of interest and draw a rectangle around the object in every image. 
Given a set of trainable images, the semi-robust object-tracking script takes in frames. take the frames and draw a box around the identified object.

run the data through the objTracking.py file. 
kalmanfilter.py has a function to calculate and predict the movement of the object. included that for better tracking of 3D data (3D FPS application)

This works on 2D images and videos the best, with 100% accuracy. when asked to track the head of a moving woman its accuracy falls 
I hope to incorporate this into a 2D video game AI bot
