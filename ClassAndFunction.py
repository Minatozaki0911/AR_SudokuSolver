import cv2
import numpy as np
import functools
from threading import Thread
from scipy.spatial import distance as dist
from keras.models import load_model

class ImageRegulate:
  def __init__(self,im):
    self.src = im
    try:
      self.contour_edges = self._contour_finder(im)
    except Exception as e:
      print(e)
      self.bError = True
      return
    self.bError = False
    self.conv_N, self.conv_M, self.warped = self._perspective_transform(im,self.contour_edges)

  def _edges_finder(self, pts):
    pts = pts.reshape(4,2)

    xSorted = pts[np.argsort(pts[:, 0]), :]

	  # grab the left-most and right-most points from the sorted
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

	  # sort the left-most coordinates according to their
	  # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # use top-left coordinate as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

	  # return the coordinates in top-left, top-right,
	  # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

  def _perspective_transform(self,img,pts):
    tl = pts[0]
    tr = pts[1]
    br = pts[2]
    bl = pts[3]

    #max width
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA),int(widthB))

    #max height
    heightA = np.sqrt( ((tl[1]-bl[1])**2) + ((tl[0]-bl[0])**2) )
    heightB = np.sqrt( ((tr[1]-br[1])**2) + ((tr[0]-br[0])**2) )
    maxHeight = max(int(heightA),int(heightB))
    
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")
    
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    N = cv2.getPerspectiveTransform(dst, pts)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return N, M, warped
  
  def _contour_finder(self,img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(7,7),3)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    

    # find contours in the thresholded image and sort them by size in descending order
    cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    
    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None
    # loop over the contours
    for c in cnts:
      if (cv2.contourArea(c) > img.shape[0]*img.shape[1]*0.1):
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
          puzzleCnt = approx
          break
    # if the puzzle contour is empty then our script could not find
  	# the outline of the Sudoku puzzle so raise an error
    if puzzleCnt is None:
      raise Exception(("Could not find Sudoku puzzle outline. ""Try debugging your thresholding and contour steps."))
    
    puzzleCnt = self._edges_finder(puzzleCnt.reshape([4,2]))
    return puzzleCnt

def setTimeOut(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('Time out!')]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout/1000)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

def extractNumbers(img, modelname):
  model = load_model(modelname)
  #img_resized = cv2.resize(img,(324,324))

  marge = 7
  case = 28 + marge*2
  taille_grille = 9 * case
  img_resized = cv2.resize(img,(taille_grille,taille_grille))
  grille_txt = []

  for y in range(9):
    ligne = ""
    for x in range(9):
      y2min = y * case + marge
      y2max = (y + 1) * case - marge
      x2min = x * case + marge
      x2max = (x + 1) * case - marge
      img_num = img_resized[y2min:y2max, x2min:x2max]
      x = img_num.reshape(1, 28, 28, 1)
      if x.sum() > 10000:
        prediction = model.predict(x)
        classes=np.argmax(prediction,axis=1)
        ligne += "{:d}".format(classes[0])
      else:
        ligne += "{:d}".format(0)
    grille_txt.append(ligne)
  print(grille_txt)
  return grille_txt

def List2Array(data):
  a = np.array(data)
  b=[]
  for i in range(9):
    string = a[i]
    arr = list(string)
    b.append(arr)
  c = np.array(b, dtype=np.uint8)
  return c

def find_next_empty(puzzle):
    # finds the next row, col on the puzzle that's not filled yet --> rep with -1
    # return row, col tuple (or (None, None) if there is none)

    # keep in mind that we are using 0-8 for our indices
    for r in range(9):
        for c in range(9): # range(9) is 0, 1, 2, ... 8
            if puzzle[r][c] == 0:
                return r, c

    return None, None  # if no spaces in the puzzle are empty (-1)

def is_valid(puzzle, guess, row, col):
    # figures out whether the guess at the row/col of the puzzle is a valid guess
    # returns True or False

    # for a guess to be valid, then we need to follow the sudoku rules
    # that number must not be repeated in the row, column, or 3x3 square that it appears in

    # let's start with the row
    row_vals = puzzle[row]
    if guess in row_vals:
        return False # if we've repeated, then our guess is not valid!

    # now the column
    # col_vals = []
    # for i in range(9):
    #     col_vals.append(puzzle[i][col])
    col_vals = [puzzle[i][col] for i in range(9)]
    if guess in col_vals:
        return False

    # and then the square
    row_start = (row // 3) * 3 # 10 // 3 = 3, 5 // 3 = 1, 1 // 3 = 0
    col_start = (col // 3) * 3

    for r in range(row_start, row_start + 3):
        for c in range(col_start, col_start + 3):
            if puzzle[r][c] == guess:
                return False

    return True

@setTimeOut(2000)
def solve_sudoku(puzzle):
    # solve sudoku using backtracking!
    # our puzzle is a list of lists, where each inner list is a row in our sudoku puzzle
    # return whether a solution exists
    # mutates puzzle to be the solution (if solution exists)
    
    # step 1: choose somewhere on the puzzle to make a guess
    row, col = find_next_empty(puzzle)

    # step 1.1: if there's nowhere left, then we're done because we only allowed valid inputs
    if row is None:  # this is true if our find_next_empty function returns None, None
        return True 
    
    # step 2: if there is a place to put a number, then make a guess between 1 and 9
    for guess in range(1, 10): # range(1, 10) is 1, 2, 3, ... 9
        # step 3: check if this is a valid guess
        if is_valid(puzzle, guess, row, col):
            # step 3.1: if this is a valid guess, then place it at that spot on the puzzle
            puzzle[row][col] = guess
            # step 4: then we recursively call our solver!
            if solve_sudoku(puzzle):
                return True
        
        # step 5: it not valid or if nothing gets returned true, then we need to backtrack and try a new number
        puzzle[row][col] = 0

    # step 6: if none of the numbers that we try work, then this puzzle is UNSOLVABLE!!
    return False

def insertNumbers(im,sudoku_problem,sudoku_ans):
  sudoku_ans_str = sudoku_ans.astype('str')
  wrapped_sudoku = im*0
  wrapped_sudoku = cv2.cvtColor(wrapped_sudoku, cv2.COLOR_GRAY2BGR)
  trans = np.array
  cell_width = wrapped_sudoku.shape[1] // 9
  cell_height = wrapped_sudoku.shape[0] // 9
  FONT = cv2.FONT_HERSHEY_DUPLEX
  FONT_COLOR = (255,255,0)
  FONT_THICCNESS = 1
  FONT_SCALE = cv2.getFontScaleFromHeight(FONT, cell_height//2, FONT_THICCNESS)

  for i in range(9):
    for j in range(9):
      if sudoku_problem[i,j] == 0:
        ans = sudoku_ans_str[i,j]
        (text_width, text_height), baseline = cv2.getTextSize(ans, FONT, FONT_SCALE, FONT_THICCNESS)
        text_x = j * cell_width + (cell_width - text_width) // 2 
        text_y = i * cell_height + (cell_height + text_height) // 2
        im_ans = cv2.putText(wrapped_sudoku, ans, 
                             (text_x, text_y), 
                             FONT, FONT_SCALE, FONT_COLOR, FONT_THICCNESS, cv2.LINE_AA)
      else:
        pass
  return im_ans