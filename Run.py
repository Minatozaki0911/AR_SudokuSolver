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
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1920)   
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
                if (queue1.empty()):
                    queue1.put(frame)

                if p1.is_alive():
                     pass
                else:
                    p1 = Process(target=Solver, args=(queue1,queue2,))
                    p1.start()  
            else:
                p1.terminate()
                p1.join()
                ans = None
                queue1.close()
                queue1 = Queue()
                queue2.close()
                queue2 = Queue()
            try:
                ans = queue2.get(timeout=0.03)
            except:
                pass

            if (ans is not None) & (sudoku.bError != True):
                N = sudoku.conv_N
                problem_ans_im = cv2.resize(ans,(sudoku.warped.shape[1], sudoku.warped.shape[0]))
                unwrapped = cv2.warpPerspective(problem_ans_im, N, (sudoku.src.shape[1], sudoku.src.shape[0]))
                output = sudoku.src + unwrapped
                output = cv2.resize(output,(640,480))
                cv2.imshow("Sudoku Solver", output)
            else:
                output = cv2.resize(frame,(640,480))
                cv2.imshow("Sudoku Solver", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):   # Hit q if you want to stop the camera (only if you have stop trying to solve the puzzle)
                break
        else:
            break

    p1.terminate()
    p1.join()
    p1.close()
    cap.release()
    cv2.destroyAllWindows()
