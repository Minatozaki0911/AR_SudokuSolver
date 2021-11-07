import Solver
import cv2

if __name__ == '__main__':
    counter = 0
    capture_path = './test.mp4'
    vid = cv2.VideoCapture(capture_path)
    
    while(vid.isOpened()):
        ret, frame = vid.read()
        counter += 1 
        if counter == cv2.CAP_PROP_FRAME_COUNT:
            counter = 0
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
        frame = cv2.bilateralFilter(frame, 5,  50, 100)

        output = Solver.solve(frame)
        print('Output updated ', counter)
        cv2.imshow('Output window', output)
        
        if cv2.waitKey(1)==ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
