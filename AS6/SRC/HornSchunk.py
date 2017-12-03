from scipy import ndimage
import numpy as np
import cv2

kernel_1 = np.array([[0.0833, 0.1666, 0.0833], [0.1666, 0.000, 0.1666], [0.0833, 0.1666, 0.0833]])

def true_val(prev, curr):
    Dx = cv2.Sobel(prev, cv2.CV_32F, 1, 0, ksize=1)
    Dy = cv2.Sobel(prev, cv2.CV_32F, 0, 1, ksize=1)
    Dt = curr - prev
    return Dx, Dy, Dt

def compute_SpatioTemporal_Derivatives(im1, im2):
	# build kernels for calculating derivatives
	kernelX = np.matrix([[-1,1],[-1,1]])*.25 #kernel for computing dx
	kernelY = np.matrix([[-1,-1],[1,1]])*.25 #kernel for computing dy
	kernelT = np.ones([2,2])*.25

	#apply the filter to every pixel using OpenCV's convolution function
	fx = cv2.filter2D(im1,-1,kernelX) + cv2.filter2D(im2,-1,kernelX)
	fy = cv2.filter2D(im1,-1,kernelY) + cv2.filter2D(im2,-1,kernelY)
	ft = cv2.filter2D(im2,-1,kernelT) + cv2.filter2D(im1,-1,-kernelT)
	return (fx,fy,ft)


def Horn_Schunk(prev, curr, alpha, itr):

    h, w = curr.shape[:2]
    flow = np.zeros((h, w, 2), np.float32)
    Dx, Dy, Dt = true_val(prev, curr)

    print "Computed Derivatives:",compute_SpatioTemporal_Derivatives(prev,curr)

    for i in range(itr):
        uAvg = ndimage.convolve(flow[:, :, 0], kernel_1, mode='constant', cval=0.0)
        vAvg = ndimage.convolve(flow[:, :, 1], kernel_1, mode='constant', cval=0.0)
        Y = alpha * alpha + np.multiply(Dx, Dx) + np.multiply(Dy, Dy)
        dyv = np.multiply(Dy, vAvg)
        dxu = np.multiply(Dx, uAvg)
        flow[:, :, 0] = uAvg - (Dx * (dxu + dyv + Dt)) / Y
        flow[:, :, 1] = vAvg - (Dy * (dxu + dyv + Dt)) / Y
    return flow


def draw_optical_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_optical_flow(img, flow):

    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':

    cam  = cv2.VideoCapture(0)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()
    print "hey"
    while True:

        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        flow = 5 * Horn_Schunk(prevgray, gray, 50, 5)
        print "Optical Floew Vector:",flow

        prevgray = gray

        cv2.imshow('flow', draw_optical_flow(gray, flow))

        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_optical_flow(cur_glitch, flow)
            cv2.imshow('glitch', cur_glitch)

        ch = 0xFF & cv2.waitKey(1)

        if ch == 'P':
            break

        if ch == ord('1'):

            show_hsv = not show_hsv
            print 'HSV flow visualization is', ['off', 'on'][show_hsv]

        if ch == ord('2'):
            show_glitch = not show_glitch

            if show_glitch:
                cur_glitch = img.copy()

            print 'glitch is', ['off', 'on'][show_glitch]

    cv2.destroyAllWindows()