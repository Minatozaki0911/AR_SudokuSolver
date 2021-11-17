import cv2
from keras.models import load_model
import numpy as np
from skimage import io 
from Solver import solve

#cv2.imshow("haiha",solve(io.imread('Ultimate-Sudoku-Collection-tri-tue.jpg')))
#cv2.imwrite(filename='image.jpg', img=solve(io.imread('HHA_5281.jpg')))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)    # HD Camera
cap.set(4, 480)
Lastproblem_ans_im= [None]
bSkip = False
while(True):
    ret, frame = cap.read() # Read the frame
    if ret == True:
        sudoku_frame, bSkip, Lastproblem_ans_im = solve(frame, bSkip, Lastproblem_ans_im) 
        #frame = overlay(frame, sudoku_frame)
        #showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600) # Print the 'solved' image
        cv2.imshow("Sudoku Solver", sudoku_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):   # Hit q if you want to stop the camera
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()