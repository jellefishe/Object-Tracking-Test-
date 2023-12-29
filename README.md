# Object-Tracking-Test-
Robust Object Tracker Given a data set

Libs used: numPy, OpenCV, Pillow/PIL

From the Umich EECS442 Computer Vision course I have learned a variety of ways to process images and videos into various types of data. I used the Kalman filter technique to identify the object of interest and draw a rectangle around the object in every image. 
Given a set of trainable images, the semi-robust object-tracking script takes in frames. take the frames and draw a box around the identified object.

run the data through the objTracking.py file. datasets: https://www.kaggle.com/datasets/kmader/videoobjecttracking
kalmanfilter.py has a function to calculate and predict the movement of the object. included that for better tracking of 3D data (3D FPS application)

This works on 2D images and videos the best, with 100% accuracy. when asked to track the head of a moving woman its accuracy falls 
I hope to incorporate this into a 2D video game AI bot

sources used:
https://medium.com/@jaems33/understanding-kalman-filters-with-python-2310e87b8f48

https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

https://arxiv.org/pdf/1204.0375.pdf

https://www.programcreek.com/python/example/110656/cv2.KalmanFilter

https://towardsdatascience.com/an-intro-to-kalman-filters-for-autonomous-vehicles-f43dd2e2004b

https://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python

https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/




