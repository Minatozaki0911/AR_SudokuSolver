import cv2
import Solver

if __name__ == '__main__':
    counter = 0
    capture_path = './test.mp4'
    if capture_path.split('.')[-1]=='mp4':
        vid = cv2.VideoCapture(capture_path)
        
        while(vid.isOpened()):
            ret, frame = vid.read()
            #counter += 1
            #if counter == cv2.CAP_PROP_FRAME_COUNT:
            #    counter = 0
            #    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
            frame = cv2.bilateralFilter(frame, 5,  50, 100)
            cv2.imshow('Input', frame)

            #output = Solver.solve(frame)
            #print('Frame ', counter, ' solved')
            #cv2.imshow('Output window', output)

            if cv2.waitKey(1)==ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()

    else:
        frame = cv2.imread(capture_path)
        output = Solver.solve(frame)
        print("Solved ez clap")
        cv2.imshow('Output window', output)

