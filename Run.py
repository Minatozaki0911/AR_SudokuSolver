import cv2
import numpy as np
from multiprocessing import Process, Queue

from ClassAndFunction import ImageRegulate
from Solver import solve

def Solver(qIn1,qOut1):
    bSkip = False
    puzzle_ans = [None]
    Lastpuzzle_ans = [None]
    while(True):
        puzzle = qIn1.get()
        
        if puzzle is None:
            qOut1.put(None)
        else:
            puzzle_ans, bSkip , Lastpuzzle_ans= solve(puzzle, bSkip, Lastpuzzle_ans) 
            qOut1.put(puzzle_ans)
            

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)    # HD Camera
    cap.set(4, 1280)
    cap.set(5,32)

    queue1 = Queue()
    queue2 = Queue()

    p1 = Process(target=Solver, args=(queue1,queue2,))
    p1.start()
 
    ans = None
    
    while(True):
        ret, frame = cap.read() # Read the frame
        frame = frame.astype(np.uint8)

        if ret == True:
            sudoku= ImageRegulate(frame)
            if sudoku.bError != True:
                queue1.put(frame)

                if p1.is_alive():
                     pass
                else:
                    ans = None
                    p1 = Process(target=Solver, args=(queue1,queue2,))
                    p1.start()  
            else:
                p1.terminate()
                queue2.close()
                queue2 = Queue()

            try:
                ans = queue2.get(timeout=0.03)
            except Exception as e:
                pass

            if (ans is not None) & (sudoku.bError != True):
                N = sudoku.conv_N
                problem = sudoku.warped
                problem = cv2.cvtColor(problem, cv2.COLOR_BGR2GRAY)
                problem = cv2.adaptiveThreshold(problem, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 8)

                problem_ans_im = cv2.resize(ans,(sudoku.warped.shape[1], sudoku.warped.shape[0]))
                unwrapped = cv2.warpPerspective(problem_ans_im, N, (sudoku.src.shape[1], sudoku.src.shape[0]))
                output = sudoku.src + unwrapped
                output = cv2.resize(output,(480,320))
                cv2.imshow("Sudoku Solver", output)
            else:
                output = cv2.resize(frame,(480,320))
                cv2.imshow("Sudoku Solver", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):   # Hit q if you want to stop the camera
                break
        else:
            break
    p1.terminate()
    p1.close()        
    cap.release()
    cv2.destroyAllWindows()
   