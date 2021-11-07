import cv2
from keras.models import load_model
import numpy as np

from ClassAndFunction import ImageRegulate
from ClassAndFunction import extractNumbers, List2Array, solveSudoku, insertNumbers

def solve(puzzle_img):

    sudoku= ImageRegulate(puzzle_img)
    problem = cv2.resize(sudoku.warped,(324,324))
    problem = cv2.cvtColor(problem, cv2.COLOR_BGR2GRAY)
    problem = cv2.adaptiveThreshold(problem, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 8)
    
    Num_list = extractNumbers(problem,"digit_model.h5")

    Num_array = List2Array(Num_list)

    problem_ans = Num_array.copy()
    if not solveSudoku(problem_ans):
        raise Exception(("Could not solve Sudoku puzzle."))
    
    problem_ans_im = insertNumbers(problem,Num_array,problem_ans)

    N = np.linalg.inv(sudoku.conv_M)
    problem_ans_im = cv2.resize(problem_ans_im,(sudoku.warped.shape[1], sudoku.warped.shape[0]))
    unwrapped = cv2.warpPerspective(problem_ans_im, N, (sudoku.src.shape[1], sudoku.src.shape[0]))
    output = cv2.cvtColor(sudoku.src,cv2.COLOR_RGB2BGR) + unwrapped
    return output
