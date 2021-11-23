import cv2
from math import sqrt

import numpy as np

from ClassAndFunction import ImageRegulate
from ClassAndFunction import extractNumbers, List2Array, solve_sudoku, insertNumbers


import cv2
from math import sqrt

import numpy as np

from ClassAndFunction import ImageRegulate
from ClassAndFunction import extractNumbers, List2Array, solve_sudoku, insertNumbers


def solve(puzzle_img, bSkip, Lastproblem_ans_im):
    sudoku= ImageRegulate(puzzle_img)

    if sudoku.bError != True:
        N = sudoku.conv_N
        problem = sudoku.warped
        problem = cv2.cvtColor(problem, cv2.COLOR_BGR2GRAY)
        problem = cv2.adaptiveThreshold(problem, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 8)
    else:
        return None, False, None
    
    if not bSkip:
        Num_list = extractNumbers(problem,"D:/Learning/Nam 4/XLA/BTL/SudokuSolver/digit_model.h5")
        Num_array = List2Array(Num_list)
        problem_ans = Num_array.copy()
        
        try:
            if not solve_sudoku(problem_ans):
                raise Exception(("Could not solve Sudoku puzzle."))
        except Exception as e:
            print(e)
            return None, False, None
        problem_ans_im = insertNumbers(problem,Num_array,problem_ans)
        return problem_ans_im, True, problem_ans_im
    else:
        return Lastproblem_ans_im, True, Lastproblem_ans_im