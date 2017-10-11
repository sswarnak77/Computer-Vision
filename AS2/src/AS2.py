import pandas as pd
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt


def ownGrayscale(image):
    return 0.299 * image[2] + 0.587 * image[1] + 0.114 * image[0]

def myImageSmoothing(image):
    kernel = np.ones((7, 7), np.float32) / 25
    dst = cv2.filter2D(image, -1, kernel)
    return dst

def XDerivative(image):
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

def YDerivative(image):
    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

def rgbSplitter(rgb):
    red, green, blue = rgb[:,:,2], rgb[:,:,1], rgb[:,:,0]
    return red,green,blue

def onTrackbar(alpha):
    alpha=alpha/255
    blur = cv2.GaussianBlur(gray, (alpha,alpha), 0)
    cv2.imshow('Smoothing Using Guassian Filtering', blur)


def nothing(x):
    pass

def grayScale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def desc():
    print "This Program has been implemented using Opencv in Python"
    print "This program is capable of performing operating on given image and if no image is given then it will access your camera to capture image and perform specified operation"
    print "Below are the given nkeys thhis program supports for different operations"
    print "i=reload original image"
    print "w=save current image"
    print "g=convert to grayscale"
    print "G=convert to grayscale using own function"
    print "s=smoothing using Gaussian kernel"
    print "S=smoothing usiong own function"
    print "d=downsample by a factor of 2 without smoothing"
    print "D=downsample by a factor of 2 with smoothing"
    print "x=x derivative filter"
    print "y=y derivative filter"
    print "m=magnitude of gradient"
    print "p=plot gradient vector"
    print "r=rotate image"


if __name__ == '__main__':

    print "Press h to get the details for this program:"
    inPut = raw_input()
    if inPut == 'h':
        desc()

    if len(sys.argv) == 2:
        filename = sys.argv[1]
        print sys.argv
        while (True):
            image = cv2.imread(filename)
            cv2.imshow('frame', image)

            if cv2.waitKey(1) & 0xFF == ord('i'):
                cv2.imshow('frame', image)

            if cv2.waitKey(1) & 0xFF == ord('w'):
                cv2.imwrite('out.jpg', image)

            if cv2.waitKey(1) & 0xFF == ord('g'):
                gray=grayScale(image)
                cv2.imshow('gray', gray)

            if cv2.waitKey(1) & 0xFF == ord('G'):
                ownGray = ownGrayscale(image)
                cv2.imshow('ownGray', ownGray)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                red,green,blue=rgbSplitter(image)
                for values, color, channel in zip((red, green, blue),('red', 'green', 'blue'), (2, 1, 0)):
                    img = np.zeros((values.shape[0], values.shape[1], 3),dtype=values.dtype)
                    img[:, :, channel] = values
                    cv2.imshow('Color Channel', img)
                    cv2.imshow('Color Channel', img)
                    cv2.imshow('Color Channel', img)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                gray = grayScale(image)
                blur = cv2.GaussianBlur(gray, (7,7), 0)
                cv2.imshow('Smoothing Using Gaussian Filtering', blur)
                g_slider = 0
                g_slider_max = 225
                cv2.createTrackbar("Control Smoothing", "Smoothing Using Guassian Filtering",g_slider,g_slider_max,nothing)
                while (1):
                    slider = cv2.getTrackbarPos('Control Smoothing', 'Smoothing Using Guassian Filtering')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    dst = cv2.GaussianBlur(gray, (7,7), 0)
                    threshold = cv2.inRange(dst, slider, g_slider_max)
                    cv2.imshow('Smoothing Using Guassian Filtering', threshold)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if cv2.waitKey(1) & 0xFF == ord('S'):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                dst = myImageSmoothing(gray)
                cv2.imshow('Smoothing Using 2D Convolution Filter',dst)
                g_slider = 0
                g_slider_max = 225
                cv2.createTrackbar("Control Smoothing", "Smoothing Using 2D Convolution Filter", g_slider, g_slider_max,nothing)
                while(1):
                    slider = cv2.getTrackbarPos('Control Smoothing', 'Smoothing Using 2D Convolution Filter')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    dst = myImageSmoothing(gray)
                    threshold = cv2.inRange(dst,slider,g_slider_max)
                    cv2.imshow('Smoothing Using 2D Convolution Filter',threshold)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if cv2.waitKey(1) & 0xFF == ord('d'):
                downsample = cv2.pyrDown(image)
                cv2.imshow('Downsample Without Smoothing', downsample)

            if cv2.waitKey(1) & 0xFF == ord('D'):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                dst = myImageSmoothing(gray)
                downsample = cv2.pyrDown(dst)
                cv2.imshow('Downsample With Smoothing',downsample)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                sobelX=XDerivative(gray)
                abs_grad_x= cv2.convertScaleAbs(sobelX)
                cv2.imshow('Convolution with an x derivative filter:',abs_grad_x)

            if cv2.waitKey(1) & 0xFF == ord('y'):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                sobelY =YDerivative(gray)
                abs_grad_y = cv2.convertScaleAbs(sobelY)
                cv2.imshow('Convolution with an y derivatyyive filter:',abs_grad_y)

            if cv2.waitKey(1) & 0xFF == ord('m'):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                sobelX=XDerivative(gray)
                sobelY=YDerivative(gray)
                dxabs = cv2.convertScaleAbs(sobelX)
                dyabs = cv2.convertScaleAbs(sobelY)
                mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
                print mag

            if cv2.waitKey(1) & 0xFF == ord('p'):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
                img = np.zeros((300, 512, 3), np.uint8)
                mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                cv2.imshow('Gradient Vector', mag)
                cv2.createTrackbar('Pixel Controller', 'Gradient Vector', 0, 255,nothing)

                while (1):
                    r = cv2.getTrackbarPos('Pixel Controller', 'Gradient Vector')
                    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
                    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
                    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                    threshold = cv2.inRange(mag, r, 255)
                    cv2.imshow('Gradient Vector', threshold)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if cv2.waitKey(1) & 0xFF == ord('r'):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rows, cols = gray.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
                dst = cv2.warpAffine(gray, M, (cols, rows))
                cv2.imshow('Rotation', dst)
                cv2.createTrackbar('Angle Controller', 'Rotation', 0, 360,nothing)
                while (1):
                    r = cv2.getTrackbarPos('Angle Controller', 'Rotation')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    rows, cols = gray.shape
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), r, 1)
                    dst = cv2.warpAffine(gray, M, (cols, rows))
                    cv2.imshow('Rotation', dst)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


    elif len(sys.argv) < 2:
        cap = cv2.VideoCapture(0)

        while (True):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('i'):
                cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('w'):
                cv2.imwrite('out.jpg', frame)

            if cv2.waitKey(1) & 0xFF == ord('g'):
                gray = grayScale(frame)
                cv2.imshow('gray', gray)

            if cv2.waitKey(1) & 0xFF == ord('G'):
                ownGray = ownGrayscale(frame)
                cv2.imshow('ownGray', ownGray)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                red, green, blue = rgbSplitter(frame)
                for values, color, channel in zip((red, green, blue), ('red', 'green', 'blue'), (2, 1, 0)):
                    img = np.zeros((values.shape[0], values.shape[1], 3), dtype=values.dtype)
                    img[:, :, channel] = values
                    cv2.imshow('Color Channel', img)


            if cv2.waitKey(1) & 0xFF == ord('s'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7,7), 0)
                cv2.imshow('Smoothing Using Guassian Filtering', blur)
                g_slider = 0
                g_slider_max = 225
                cv2.createTrackbar("Control Smoothing", "Smoothing Using Guassian Filtering",g_slider,g_slider_max,nothing)
                while (1):
                    slider = cv2.getTrackbarPos('Control Smoothing', 'Smoothing Using Guassian Filtering')
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    dst = cv2.GaussianBlur(gray, (7,7), 0)
                    threshold = cv2.inRange(dst, slider, g_slider_max)
                    cv2.imshow('Smoothing Using Guassian Filtering', threshold)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if cv2.waitKey(1) & 0xFF == ord('S'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dst = myImageSmoothing(gray)
                cv2.imshow('Smoothing Using 2D Convolution Filter',dst)
                g_slider = 0
                g_slider_max = 225
                cv2.createTrackbar("Control Smoothing", "Smoothing Using 2D Convolution Filter", g_slider, g_slider_max,nothing)
                while(1):
                    slider = cv2.getTrackbarPos('Control Smoothing', 'Smoothing Using 2D Convolution Filter')
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    dst = myImageSmoothing(gray)
                    threshold = cv2.inRange(dst,slider,g_slider_max)
                    cv2.imshow('Smoothing Using 2D Convolution Filter',threshold)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if cv2.waitKey(1) & 0xFF == ord('d'):
                downsample = cv2.pyrDown(frame)
                cv2.imshow('Downsample Without Smoothing', downsample)

            if cv2.waitKey(1) & 0xFF == ord('D'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dst = myImageSmoothing(gray)
                downsample = cv2.pyrDown(dst)
                cv2.imshow('Downsample With Smoothing',downsample)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sobelX=XDerivative(gray)
                abs_grad_x= cv2.convertScaleAbs(sobelX)
                cv2.imshow('Convolution with an x derivative filter:',abs_grad_x)

            if cv2.waitKey(1) & 0xFF == ord('y'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sobelY =YDerivative(gray)
                abs_grad_y = cv2.convertScaleAbs(sobelY)
                cv2.imshow('Convolution with an y derivatyyive filter:',abs_grad_y)

            if cv2.waitKey(1) & 0xFF == ord('m'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sobelX=XDerivative(gray)
                sobelY=YDerivative(gray)
                dxabs = cv2.convertScaleAbs(sobelX)
                dyabs = cv2.convertScaleAbs(sobelY)
                mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
                print mag

            if cv2.waitKey(1) & 0xFF == ord('p'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
                img = np.zeros((300, 512, 3), np.uint8)
                mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                cv2.imshow('Gradient Vector', mag)
                cv2.createTrackbar('Pixel Controller', 'Gradient Vector', 0, 255, nothing)

                while (1):
                    r = cv2.getTrackbarPos('Pixel Controller', 'Gradient Vector')
                    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
                    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
                    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                    threshold = cv2.inRange(mag, r, 255)
                    cv2.imshow('Gradient Vector', threshold)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if cv2.waitKey(1) & 0xFF == ord('r'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rows, cols = gray.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
                dst = cv2.warpAffine(gray, M, (cols, rows))
                cv2.imshow('Rotation', dst)
                cv2.createTrackbar('Angle Controller', 'Rotation', 0, 360, nothing)
                while (1):
                    r = cv2.getTrackbarPos('Angle Controller', 'Rotation')
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rows, cols = gray.shape
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), r, 1)
                    dst = cv2.warpAffine(gray, M, (cols, rows))
                    cv2.imshow('Rotation', dst)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()




