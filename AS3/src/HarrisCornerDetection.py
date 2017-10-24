import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pyp


#5,1000000,5,5

def HarrisCorner(image,window_size,hCons,threshold,sigmaX,sigmaY):

    Blur = cv2.GaussianBlur(image, (sigmaX,sigmaY), 0)

    Ix = cv2.Sobel(Blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(Blur, cv2.CV_64F, 0, 1, ksize=5)

    Ix, Iy = np.gradient(np.array(Blur, dtype=np.float))

    Ixx = Ix ** 2
    Ixy = Iy* Ix
    Iyy = Iy ** 2

    height = image.shape[0]
    width = image.shape[1]

    R = [[0 for _ in range(width)] for _ in range(height)]

    foundCorners = []
    BetterCorners=[]

    output_img = image.copy()
    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)
    offset = window_size / 2

    for i in range(offset, height - offset):

        for j in range(offset, width - offset):

            IxxWindow = Ixx[i - offset:i + offset + 1, j - offset:j + offset + 1]
            IxyWindow = Ixy[i - offset:i + offset + 1, j - offset:j + offset + 1]
            IyyWindow = Iyy[i - offset:i + offset + 1, j - offset:j + offset + 1]

            Sxx = IxxWindow.sum()
            Sxy = IxyWindow.sum()
            Syy = IyyWindow.sum()

            deter = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy

            r = deter - hCons * (trace ** 2)
            R[i][j] = r
            if r > threshold:
                foundCorners.append([j, i, r])
                #cv2.rectangle(output_img, (j, i), (j, i), (0, 0, 255), 3)

    print "Number of corners before Suppression: ", len(foundCorners)

    for corner in foundCorners:
        x, y, r = corner[0], corner[1], corner[2]
        if not (x >= 9 and y >= 9 and y < height - 9 and x < width - 9):
            continue
        if r > R[y + 1][x + 1]:
            if r > R[y + 1][x]:
                if r > R[y + 1][x - 1]:
                    if r > R[y][x - 1]:
                        if r > R[y][x + 1]:
                            if r > R[y - 1][x + 1]:
                                if r > R[y - 1][x]:
                                    if r > R[y - 1][x - 1]:
                                        BetterCorners.append(corner)
                                        cv2.rectangle(output_img,(x,y),(x,y),(0, 0, 255), 3)

    print "Number of corners after Suppression: ", len(BetterCorners)
    return output_img,BetterCorners



if __name__ == '__main__':

    sigmaX = int(input("Enter VarianceX for Gaussian: "))
    sigmaY = int(input("Enter VarianceY for Gaussian : "))
    window_size = int(input("Enter Window Size : "))
    threshold = int(input("Enter Threshold for Corner Detection : "))
    hCons = int(input("Enter Harris Constant for Corner Detection : "))

    image1 = cv2.imread('chair1.jpg')
    image2 = cv2.imread('chair2.jpg')

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    finalImg1,cornerList1 = HarrisCorner(gray1,window_size,hCons,threshold,sigmaX,sigmaY)
    finalImg2,cornerList2 = HarrisCorner(gray2,window_size,hCons,threshold,sigmaX,sigmaY)

    cv2.imwrite("finalImg1.png", finalImg1)
    cv2.imwrite("finalImg2.png", finalImg2)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(gray1, kp1, gray2, kp2, matches, None, flags=2)
    cv2.imwrite("MatchedImg1.png", img3)

    plt.imshow(img3), plt.show()






















