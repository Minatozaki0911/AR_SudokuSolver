import cv2
from math import sqrt

import numpy as np
from numpy.linalg import inv

from ClassAndFunction import ImageRegulate
from ClassAndFunction import extractNumbers, List2Array, solveSudoku, insertNumbers


def solve(puzzle_img, bSkip, Lastproblem_ans_im):
    sudoku= ImageRegulate(puzzle_img)
    if sudoku.bError != True:
        N = sudoku.conv_N
        problem = sudoku.warped
        problem = cv2.cvtColor(problem, cv2.COLOR_BGR2GRAY)
        problem = cv2.adaptiveThreshold(problem, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 8)
    else:
        output = sudoku.src
        return output, False, None
    
    Num_list = extractNumbers(problem,"digit_model.h5")

    Num_array = List2Array(Num_list)
    
    if not bSkip:
        problem_ans = Num_array.copy()
        try:
            if not solveSudoku(problem_ans):
                raise Exception(("Could not solve Sudoku puzzle."))
        except Exception as e:
            print(e)
            output = sudoku.src
            return output, False, None
        problem_ans_im = insertNumbers(problem,Num_array,problem_ans)

        problem_ans_im = cv2.resize(problem_ans_im,(sudoku.warped.shape[1], sudoku.warped.shape[0]))
        Lastproblem_ans_im = problem_ans_im
        unwrapped = cv2.warpPerspective(problem_ans_im, N, (sudoku.src.shape[1], sudoku.src.shape[0]))
        output = sudoku.src + unwrapped
        return output, True, Lastproblem_ans_im
    else:
        problem_ans_im = Lastproblem_ans_im

        problem_ans_im = cv2.resize(problem_ans_im,(sudoku.warped.shape[1], sudoku.warped.shape[0]))
        unwrapped = cv2.warpPerspective(problem_ans_im, N, (sudoku.src.shape[1], sudoku.src.shape[0]))
        output = sudoku.src + unwrapped
        return output, True, Lastproblem_ans_im
    
        

    
    
