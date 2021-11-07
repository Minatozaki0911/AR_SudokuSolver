import cv2
import numpy as np
from scipy.spatial import distance as dist
from keras.models import load_model

class ImageRegulate:
  def __init__(self,im):
    self.src = im
    self.contour_edges = self._contour_finder(im)
    self.conv_M, self.warped = self._perspective_transform(im,self.contour_edges)

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
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return M, warped
  
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

def extractNumbers(img, modelname):
  model = load_model(modelname)
  img_resized = cv2.resize(img,(324,324))

  marge = 4
  case = 36
  taille_grille = 9 * case

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

def solveSudoku(board):
  find = _find_empty(board)
  if not find:
    return True
  else:
    row, col = find
  for i in range(1,10):
    if _valid(board, i, (row, col)):
      board[row][col] = i
      if solveSudoku(board):
        return True
      board[row][col] = 0
  return False

def _find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col
    return None

def _valid(board, num, pos):
    # Check row
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False
    # Check column
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False
    return True

def insertNumbers(im,sudoku_problem,sudoku_ans):
  sudoku_ans_str = sudoku_ans.astype('str')
  wrapped_sudoku = 255 - im
  wrapped_sudoku = cv2.cvtColor(wrapped_sudoku, cv2.COLOR_GRAY2BGR)
  trans = np.array
  cell_width = wrapped_sudoku.shape[1] // 9
  cell_height = wrapped_sudoku.shape[0] // 9
  FONT = cv2.FONT_HERSHEY_DUPLEX
  FONT_COLOR = (128,128,0)
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