import numpy as np
import cv2
import glob
import math
import random
import copy
import configparser

# termination criteria
import sys


def nonPlanarFeatures(imagePath):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:7].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    img = cv2.imread(imagePath)

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(grayimg, 72, 0.05, 25)
    i =0
    corners = np.float32(corners)
    with open('ncc-worldPt.txt', 'w') as wp, open('ncc-imagePt.txt', 'w') as ip:

        for item in corners:
            x, y = item[0]
            writeTofile = str(x) + ' ' + str(y) + '\n'
            ip.write(writeTofile)
            if (i < 24):
                writeTofile = str(x) + ' ' + str(y) + ' ' + str(0) + '\n'
            elif (i>=24 and i<48):
                writeTofile = str(x) + ' ' + str(0) + ' ' + str(y) + '\n'
            else:
                writeTofile = str(0) + ' ' + str(x) + ' ' + str(y) + '\n'
            wp.write(writeTofile)
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(img, str(i),  (x,y), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            i = i+1
            cv2.imshow("Corner Detection" , img)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def calibrateCamera(csvfilepath):
    pointList2D = []
    pointList3D = []
    with open(csvfilepath) as file:
        line = file.readline()
        while line:
            points = line.strip().split(',')
            points_3D = []
            points_3D.append(float(points[0]))
            points_3D.append(float(points[1]))
            points_3D.append(float(points[2]))
            pointList3D.append(points_3D)

            points_2D = []
            points_2D.append(float(points[3]))
            points_2D.append(float(points[4]))
            pointList2D.append(points_2D)
            line = file.readline()
    numberOfpoints = len(pointList2D)
    print(numberOfpoints)
    A = []

    for i in range(0, numberOfpoints):
        col = 0
        temp_A = []
        temp_A.append(pointList3D[i][0])
        temp_A.append(pointList3D[i][1])
        temp_A.append(pointList3D[i][2])
        temp_A.append(1.0)
        temp_A.append(0.0)
        temp_A.append(0.0)
        temp_A.append(0.0)
        temp_A.append(0.0)
        temp_A.append( (-1) * pointList2D[i][0] * pointList3D[i][0])
        temp_A.append((-1) * pointList2D[i][0] * pointList3D[i][1])
        temp_A.append((-1) * pointList2D[i][0] * pointList3D[i][2])
        temp_A.append((-1) * pointList2D[i][0])

        A.append(temp_A)
        temp_A = []
        temp_A.append(0.0)
        temp_A.append(0.0)
        temp_A.append(0.0)
        temp_A.append(0.0)
        temp_A.append(pointList3D[i][0])
        temp_A.append(pointList3D[i][1])
        temp_A.append(pointList3D[i][2])
        temp_A.append(1.0)
        temp_A.append((-1) * pointList2D[i][1] * pointList3D[i][0])
        temp_A.append((-1) * pointList2D[i][1] * pointList3D[i][1])
        temp_A.append((-1) * pointList2D[i][1] * pointList3D[i][2])
        temp_A.append((-1) * pointList2D[i][1])

        A.append(temp_A)
    A = np.array(A)
    A.reshape(numberOfpoints*2, 12)
    U, S, V = np.linalg.svd(A, full_matrices=True)
    V_transpose = V.T
    M = []
    row = 0
    b = []
    for i in range(0,3):
        Projection_Matrix = []
        for j in range(0,4):
            Projection_Matrix.append(V_transpose[row][11])
            if j==3:
                b.append(V_transpose[row][11])
            row += 1

        M.append(Projection_Matrix)
    return getcameraParams(M,b,pointList2D,pointList3D)

def getcameraParams(M,b,pointList2D,pointList3D):
    M = np.array(M)
    M.reshape(3,4)
    m1 = np.array(M[0])
    m2 = np.array(M[1])
    m3 = np.array(M[2])
    actualPoint2DList = []

    #Finding Actual Points and Mean Square Error
    count = 0
    mse = 0
    for points in pointList3D:
         actualPoint2D = []
         points.append(1)
         points = np.array(points)
         xpoint2D = float(np.dot((m1.T),points))/float(np.dot((m3.T),points))
         ypoint2D = float(np.dot((m2.T),points))/float(np.dot((m3.T),points))
         actualPoint2D.append((xpoint2D,ypoint2D))
         actualPoint2DList.append(actualPoint2D)

         mse += pow((pointList2D[count][0] - xpoint2D ),2) + pow((pointList2D[count][1] - ypoint2D),2)
         count +=1

    mse = mse/len(pointList2D)

    a1 = (m1[:3]).T
    a2 = (m2[:3]).T
    a3 = (m3[:3]).T

    #Solving for equation parameters
    z = (a3 * a3).sum()
    pDet = 1 / (z ** 0.5)

    print "Pdeter",pDet

    u0 = float(np.dot((pDet * pDet), np.dot(a1,a3)))
    u0 = round(u0,2)

    v0 = float((pDet * pDet) * np.dot(a2, a3))
    v0 = round(v0, 2)

    print "u0,v0",u0,v0

    alphav = math.sqrt(float(np.dot(pow(pDet ,2), np.dot(a2,a2)) - v0*v0))
    alphav = round(alphav,2)

    s = (float(alphav* np.dot(np.cross(a1,a3), np.cross(a2,a3))))
    s = math.ceil(s)

    alphau = (float(pow(pDet, 2)*np.dot(a2,a2) - pow(s,2) - pow(v0, 2)) ** 0.5)
    alphau = round(alphau,2)

    print "alphau,alphav",alphau,alphau

    K = np.array([[alphau, s, u0],[0,alphav,v0],[0,0,1]]).reshape(3,3)

    print "K",K

    e = np.sign(b[2])

    inv = np.linalg.inv(K)
    T = e * pDet * np.dot(inv,b)

    r3= e * pDet* a3
    r1 = np.cross((alphav*a2),a3)
    r2= np.cross(r1,r3)
    R = [r1,r2,r3]
    R= np.array(R).reshape(3,3)

    return mse, K, T, R

def Ransac(k_max, n, count , totalset ,l,pointList):

    p = 0.99 #Given
    k = math.ceil(np.log(1-0.99)/np.log(1-math.pow(0.5,n))) #Calculating number of trials

    while(count<k):
        count = count +1
        if (count>k_max):
            break
        else:
            test = random.sample(pointList,n)
            mat, worldpoint, imagepoint = getMatrix(test)

            projection_matrix = projection(mat)
            threshold = getThreshold(projection_matrix, test)

            new_s, length = getInliers(projection_matrix, pointList, threshold)
            w = (len(new_s)/float(len(pointList)))
            k = math.ceil(np.log(1-0.99)/np.log(1-math.pow(w,n)))

            if(length> 6):
                all_S.append(new_s)
                all_len.append(length)

    b = totalset[np.argmax(l)]
    matt, worldpoint, imagepoint = getMatrix(b)
    Matrix_M = projection(matt)
    b1 = []
    for values in Matrix_M:
        b1.append(values[3])
    return Matrix_M, b1

#Get Projection Matrix
def getMatrix(data):
    data = np.array(data)
    finalmatrix = []
    wpoint = []
    ipoint = []
    for i in data:
        X_,Y_,Z_,x,y = i.split(' ')
        X_ = float(X_)
        Y_ = float(Y_)
        Z_ = float(Z_)
        x = float(x)
        y = float(y)

        a = X_,Y_,Z_,1,0,0,0,0,-x*X_,-x*Y_,-x*Z_,-x
        b = 0,0,0,0,X_,Y_,Z_,1,-y*X_,-y*Y_,-y*Z_,-y
        c = X_,Y_,Z_,1
        d= x,y
        temp =[]
        temp.extend(a)
        finalmatrix.append(temp)
        temp = []
        temp.extend(b)
        finalmatrix.append(temp)
        wpoint.append(c)
        ipoint.append(d)


    return finalmatrix,wpoint,ipoint

def projection(M):
    U, S, V = np.linalg.svd(M)
    m = V.T[:,-1].reshape(3,4)
    return m

def getThreshold(projection_matrix, test):
    a,b,c = getMatrix(test)
    distance = []
    for i in range(len(test)):

        ui,vi,wi = np.matmul(projection_matrix,b[i])
        if not ui == 0:
            u = ui/wi
            v = vi/wi
        else:
            u = ui
            v = vi
        d,e = c[i]
        dist = math.sqrt(math.pow(d-u,2) + math.pow(e-v,2))
        distance.append(dist)
    return (1.5*(np.median(distance)))

def getInliers(M,data,thres):
    a, wPoint, iPoint = getMatrix(data)

    inlierP = []
    for i in range(len(data)):
        ui,vi,wi=np.matmul(M, wPoint[i])
        if not wi == 0:
            u = float(ui)/float(wi)
            v=float(vi)/float(wi)
        else:
            u = ui
            v = vi
        a,b = iPoint[i]
        dist = math.sqrt(math.pow(a-u,2) + math.pow(b-v,2))

        if(dist < thres):
            inlierP.append(data[i])
    return inlierP, len(inlierP)

#Feature Extraction
print("Feature Extraction:")
imagePath = 'chess_3d.png'
nonPlanarFeatures(imagePath)


#Camera Calibration
print("Non planar camera calibration")
imgPoints = 'ncc-imagePt.txt'
objPoints = 'ncc-worldPt.txt'

imagepoints_list = []
objectpoints_list = []
with open(imgPoints) as i_points, open(objPoints) as o_points:
    line_i = i_points.readline()
    line_o = o_points.readline()
    with open('CalibrateFile.csv', 'w') as file:
        while line_i and line_o:

            imagepoints =line_i.split()
            objectpoints = line_o.split()
            if(len(imagepoints) == 2 and len(objectpoints) == 3):
                writetocsv = objectpoints[0] + ',' + objectpoints[1] + ',' + objectpoints[2] + ',' +  imagepoints[0] + ',' + imagepoints[1]
                file.write(writetocsv)
                file.write('\n')
            line_i = i_points.readline()
            line_o = o_points.readline()

csvfilepath = 'CalibrateFile.csv'

print("Estimated Parameters:")
mse,K, T, R = calibrateCamera(csvfilepath)
print("Mean Square Error", mse)
print("Intrinsic Parameter- \nK:", K)
print("Extrinsic Parameters - \nT:", T , "\nR:", R)

##Ransac
print("Ransac Implementation")
noiseImgPoints = 'Noise1/ncc-noise-0-imagePt.txt'
noiseObjPoints = 'Noise1/ncc-worldPt.txt'

content = set()
noise3DPointList = []
noise2DPointList = []
with open(noiseImgPoints) as i_points, open(noiseObjPoints) as o_points:
    line_i = i_points.readline()
    line_o = o_points.readline()
    with open('csvfile.csv', 'w') as file:
        while line_i and line_o:

            imagepoints =line_i.split()
            objectpoints = line_o.split()
            if(len(imagepoints) == 2 and len(objectpoints) == 3):
                writetocsv = objectpoints[0] + ' ' + objectpoints[1] + ' ' + objectpoints[2] + ' ' +  imagepoints[0] + ' ' + imagepoints[1]
                noise3DPointList.append([float(objectpoints[0]) ,float(objectpoints[1]) ,float(objectpoints[2])])
                noise2DPointList.append([float(imagepoints[0]),float(imagepoints[1])])
                content.add(writetocsv)
            line_i = i_points.readline()
            line_o = o_points.readline()

config = configparser.ConfigParser()
config.sections()
config.read('RANSAC.config')
n = int(config['RANSAC_PARAMETERS']['N'])
max_k = int(config['RANSAC_PARAMETERS']['MAX_K'])
count = 0
all_S=[]
all_len=[]
M, b1 = Ransac(max_k,n,count,all_S,all_len,list(content))
mseN,KN, TN, RN = getcameraParams(M, b1,noise2DPointList,noise3DPointList)
print("Estimated Parameters:")
print("Mean Square Error", mseN)
print("Intrinsic Parameter- K:", KN)
print("Extrinsic Parameters - T:", TN , "\nR:", RN)


