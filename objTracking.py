import cv2
import numpy as np
from pdb import set_trace
import motmetrics as mm
from PIL import Image
import sys
import statistics
from KalmanFilter import KalmanFilter

# Function to read image. Change greyscale to True for greyscale image
def read_img(path, greyscale=False):
    img = Image.open(path)
    if greyscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    return np.array(img).astype(np.uint8)

def main():
    # Select dataset. For my test project, either 'girl' or 'head_motion'. email me if you want to test on these 2 image fodlers
    dataset = sys.argv[1]

    # Number of frames. 
    num_frames = int(sys.argv[2])

    # The following is my implementation of Kalman Filter, go to kalman.py to see documentation
    kf = KalmanFilter()

    # The following is cv2 implementation of Kalman Filter. Simply uncomment lines 30-33 and comment out line 27 to use cv2 
    #kf = cv2.KalmanFilter(4,2)
    #kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    #kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    #kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

    # Detected are the detected coordinates
    # Corrected are the corrected coordinates using the Kalman Filter algorithm
    # Predicted are the predicted coordinates using the Kalman Filter algorithm
    detected = np.array((2, 1), np.float32)
    corrected = np.array((2, 1), np.float32)
    predicted = np.array((2, 1), np.float32)
    
    # Open the ground truth file, which hold the coordinates for the position of the object email me for these files
    groundtruth_file = open(f'{dataset}/groundtruth.txt', 'r')
    groundtruth_file = groundtruth_file.readlines()
    
    # Lists used later to calculate median, mean, standard deviation
    detected_list = []
    corrected_list = []
    predicted_list = []

    # Loop over each frame
    for i in range(num_frames):
        # The names of frames in the dataset vary. For other datasets, this should be changed.
        if dataset == 'girl':
            frame = read_img(f'girl/img{i:03}.png')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif dataset == 'head_motion':
            frame = read_img(f'head_motion/input_{i:06}.png')

        # Detect faces using cv2
        face_cascade = cv2.CascadeClassifier('facedetector.xml')
        faces = face_cascade.detectMultiScale(frame, 1.1, 4)

        # If face is detected, continue using Kalman. Otherwise, go to the next loop iteration until a face is detected
        if isinstance(faces, np.ndarray):
            # Draw a rectangle around the detected face. 
            # w and h correspond to the width and height of the detected face. x and y are the top left of the detected face
            # Currently, we only detect 1 face. More than 1 face can be detected, but implementation would need to change to apply Kalman to both
            (x, y, w, h) = faces[0]
            # Red rectangle is detected position
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            detected[0] = x
            detected[1] = y

            # Use Kalman Filter to get the corrected coordinates
            corrected = kf.correct(detected)  
            
            # Use Kalman Filter to predict new position
            predicted = kf.predict()           
            
            # The green rectangle is the correct position
            cv2.rectangle(frame, (int(corrected[0]), int(corrected[1])), (int(corrected[0] + w), int(corrected[1] + h)), (0,255,0), 2)
            
            # The blue rectangle is the predicted position
            cv2.rectangle(frame, (int(predicted[0]), int(predicted[1])), (int(predicted[0] + w), int(predicted[1] + h)), (0,0,255), 2)

            # parse ground-truth file
            gt = groundtruth_file[i]
            gt_x, gt_y, gt_w, gt_h = gt.split(",")[0], gt.split(",")[1], gt.split(",")[2], gt.split(",")[3]
            
            # Some lines on the ground-truth file have no information. In this case, exit loop.
            if gt_x == 'NaN' or gt_y == 'NaN' or gt_w == 'NaN' or gt_h == 'NaN':
                cv2.destroyAllWindows()
                break

            # The following are 4 matrices created to compute the distance
            gt_mat = np.array((int(gt_x), int(gt_y)))
            detected_mat = np.array((detected[0], detected[1]))
            corrected_mat = np.array((corrected[0][0], corrected[1][0]))
            predicted_mat = np.array((predicted[0][0], predicted[1][0]))

            # Euclidean distance between ground-truth and detected/corrected/predicted
            detected_distace = np.linalg.norm(gt_mat - detected_mat)
            corrected_distance = np.linalg.norm(gt_mat - corrected_mat)
            predicted_distance = np.linalg.norm(gt_mat - predicted_mat)

            # append to corresponding list. Will be used later for calculation
            detected_list.append(detected_distace)
            corrected_list.append(corrected_distance)
            predicted_list.append(predicted_distance)

        # pop-up window for the image    
        cv2.imshow('image', frame)
       
       # Press 'q' to exit pop-up window
        if cv2.waitKey(2) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    d_median = statistics.median(detected_list)
    d_mean = statistics.mean(detected_list)
    d_std_deviation = statistics.stdev(detected_list)

    c_median = statistics.median(corrected_list)
    c_mean = statistics.mean(corrected_list)
    c_std_deviation = statistics.stdev(corrected_list)

    p_median = statistics.median(predicted_list)
    p_mean = statistics.mean(predicted_list)
    p_std_deviation = statistics.stdev(predicted_list)

    print("Detected median: ", d_median)
    print("Detected mean: ", d_mean)
    print("Detected standard deviation: ", d_std_deviation)

    print("Corrected median: ", c_median)
    print("Corrected mean: ", c_mean)
    print("Corrected standard deviation: ", c_std_deviation)

    print("Predicted median: ", p_median)
    print("Predicted mean: ", p_mean)
    print("Predicted standard deviation: ", p_std_deviation)

if __name__ == "__main__":
    main()
