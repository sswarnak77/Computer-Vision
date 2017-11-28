import numpy as np, cv2
from scipy import linalg
from pylab import *
from numpy import *
import pylab

refPt = []
cropping = False
inputArray=[]


def displayImageSidebySide(img1,img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRcAY2BGR)
    return vis

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        print refPt
        cropping = True


    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        print x,y
        cropping = False
        inputArray.append((x,y))
        print "Input Array Length",len(inputArray)

        # draw a rectangle around the region of interest

        #cv2.rectangle(leftImage, refPt[0], refPt[1], (0, 255, 0), 2)
        #cv2.rectangle(rightImage, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("leftImage", leftImage)
        #cv2.imshow("rightImage", rightImage)

def getMouseInput(leftImage,rightImage):

    clone1 = leftImage.copy()
    clone2 = rightImage.copy()

    cv2.namedWindow('leftImage')
    cv2.setMouseCallback('leftImage', click_and_crop)
    cv2.namedWindow('rightImage')
    cv2.setMouseCallback('rightImage', click_and_crop)

    while True:

        # display the image and wait for ca keypress

        cv2.imshow("leftImage", leftImage)
        cv2.imshow("rightImage", rightImage)

        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            img1 = clone1.copy()
            img2 = clone2.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # if there are two reference points, then crop the region of interest

    # close all open windows
    cv2.destroyAllWindows()

def compute_fundamental(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    A = zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

    # compute linear least square solution
    U, S, V = linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = linalg.svd(F)
    S[2] = 0
    F = dot(U, dot(diag(S), V))

    return F / F[2, 2]


def compute_fundamental_normalized(x1, x2):
  '''Computes the fundamental matrix from corresponding points x1, x2 using
  the normalized 8 point algorithm.'''

  n = x1.shape[1]
  if x2.shape[1] != n:
    raise ValueError('Number of points do not match.')

  # normalize.
  x1 = x1 / x1[2]
  mean_1 = np.mean(x1[:2], axis=1)
  S1 = np.sqrt(2) / np.std(x1[:2])

  T1 = np.array([[S1, 0, -S1 * mean_1[0]],
                    [0, S1, -S1 * mean_1[1]],
                    [0, 0, 1]])

  x1 = np.dot(T1, x1)

  x2 = x2 / x2[2]
  mean_2 = np.mean(x2[:2], axis=1)
  S2 = np.sqrt(2) / np.std(x2[:2])

  T2 = np.array([[S2, 0, -S2 * mean_2[0]],
                    [0, S2, -S2 * mean_2[1]],
                    [0, 0, 1]])
  x2 = np.dot(T2, x2)

  F = compute_fundamental(x1, x2)

  # denormalize.
  F = np.dot(T1.T, np.dot(F, T2))
  return F / F[2, 2]


def compute_right_epipole(F):
    """ Computes the (right) epipole from a
        fundamental matrix F"""

    # return null space of F (Fx=0)
    U, S, V = linalg.svd(F)
    e = V[-1]
    return e / e[2]

def compute_left_epipole(F):
    """ Computes the (left) epipole from a
        fundamental matrix F"""

    # return null space of F (Fx=0)
    U, S, V = linalg.svd(F)
    e = V[-1]
    return e / e[2]


def plot_epipolar_line(im, F,x, epipole=None, show_epipole=True):

    """ Plot the epipole and epipolar line F*x=0
        in an image. F is the fundamental matrix
        and x a point in the other image."""
    m, n = im.shape[:2]
    line = dot(F, x)

    # epipolar line parameter and values
    t = linspace(0, n, 100)
    lt = array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt >= 0) & (lt < m)
    plot(t[ndx], lt[ndx], linewidth=2)

    if show_epipole:
        if epipole is None:
            epipole = compute_right_epipole(F)
        plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')

    return line

def plot_epipolar_line1(image1, F, x,y, epipole=None, show_epipole=True):

    point=np.array([[x],[y],[1]])
    line=np.transpose(F).dot(point)
    h,w=image1.shape[0],image1.shape[0]
    print h,w
    min=8000000000
    for i in range(h):
        for c in range(w):
            p1=np.array([c,i,1])
            if min>np.abs(p1.dot(line)):
                min = np.abs(p1.dot(line))
            if np.abs(p1.dot(line))<0.000001:
                cv2.circle(image1,(c,i),1,1,1)
        cv2.imshow("leftInfo",image1)
        cv2.waitKey(1)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

    c = img1.shape[1]

    for r, pt1, pt2 in zip(lines, pts1, pts2):

        x0, y0 = map(int, [0, -r[2] /r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img1, (x0, y0), (x1, y1), (255,0,0), 1)
        cv2.circle(img1, tuple(pt1), 5, (255,0,0), -1)
        cv2.circle(img2, tuple(pt2), 5, (255,0,0), -1)
    return img1,img2


if __name__ == '__main__':
    leftImage='left.jpg'
    rightImage='right.jpg'

    leftImage=cv2.imread(leftImage,0)
    rightImage=cv2.imread(rightImage,0)

    getMouseInput(leftImage,rightImage)

    x1=inputArray[:8]
    x2=inputArray[8:]

    #output_image=displayImageSidebySide(image1,image2)
    #x1=[(300, 214),(284, 238),(266, 203),(177, 211),(221, 236),(252, 224),(352, 204),(144, 241)]
    #x2=[(301, 225),(271, 231),(272, 213),(195, 217),(244, 230),(242, 214),(345, 218),(183, 204)]

    x1 = np.array(x1)
    x2 = np.array(x2)

    F = compute_fundamental(x1, x2)
    right_epipole = compute_right_epipole(F)
    left_epipole = compute_right_epipole(F.T)
    print "Fundanental Matrix:",F
    print "Right Epipole:",right_epipole
    print "Left Epipole:",left_epipole

    getMouseInput(leftImage, rightImage)
    x,y=inputArray[len(inputArray)-1]

    print x,y

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = plot_epipolar_line(leftImage,F,[x,y,1])
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(leftImage, rightImage, lines1, x1, x2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = plot_epipolar_line(rightImage,F,[x,y,1])
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(rightImage, leftImage, lines2, x2, x1)

    #img1,img2=drawlines(leftImage,rightImage,lines,x1,x2)

    cv2.imwrite('drawingOnLeft.png', img5)
    cv2.imwrite("drawingOnRight.png",img3)
