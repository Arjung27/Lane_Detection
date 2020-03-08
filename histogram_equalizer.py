import os
import sys
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def hist_equalize(img):

    # img2 = cv2.GaussianBlur(img, (5,5), 0)
    # img2 = adjust_gamma(img2, 10)
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    N = img.shape[0]*img.shape[1]
    cdf_m = cdf_m*255/N
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]
    cv2.imshow("image", img2)
    cv2.waitKey(0)
    # plt.imshow(img2)
    # plt.show()

def hough_tf(img, img_):

    # img_ = adjust_gamma(img_, 2)
    # cv2.imshow(" s", img_)
    # cv2.waitKey(0)
    # img_= cv2.GaussianBlur(img_,(49,49),0)
    cv2.imshow("b", img)
    cv2.waitKey(0)
    lower_color_bound = (0, 150, 180)
    upper_color_bound = (200, 180, 190)
    # _, img_ = cv2.threshold(img_, 200, 255, cv2.THRESH_BINARY)
    img_ = cv2.inRange(img, lower_color_bound, upper_color_bound)
    cv2.imshow("a", img_)
    cv2.waitKey(0)
    # img_ = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_, 20, 20, apertureSize=3)
    edges = cv2.GaussianBlur(edges,(35,35),0)
    # edges = adjust_gamma(edges, 2)
    # edges = cv2.Sobel(img_,cv2.CV_64F,1,0,ksize=5)
    cv2.imshow(" s", edges)
    cv2.waitKey(0)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 10, 10)
    # print(lines.shape)
    for x1, y1, x2, y2, in lines[:,0,:]:
        # print (x1, y1, x2, y2)
        if np.abs((y2 - y1)/(x2 - x1)) < 2:
            continue
        cv2.line(img, (x1,y1), (x2, y2), (0,255,0), 2)
    # exit(-1)
    # print(lines)
    # lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    # for rho, theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = abs(int(x0 + 1000*(-b)))
    #     y1 = abs(int(y0 + 1000*(a)))
    #     x2 = abs(int(x0 - 1000*(-b)))
    #     y2 = abs(int(y0 - 1000*(a)))
    #     print (x1, y1, x2, y2)
    #     cv2.line(img, (x1,y1), (x2, y2), (0,255,0), 2)
    cv2.imshow("lines",img)
    cv2.waitKey(0)
    # exit(-1)
    

    return    

if __name__ == '__main__':

    if sys.argv[2] == 'mp4':
        video_file = sys.argv[1]
        cap = cv2.VideoCapture(video_file)

        while (cap.isOpened()):

            ret, img = cap.read()
            if ret == False:
                break

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("test homography", img)
            # hist_equalize(img)
            src_pt = np.array([[407, 615], [886, 615],[312, 677],[962, 677] ])
            dst_pt = np.array([[150, 510], [1080, 510], \
                    [150, 700], [1080, 700]])

            H, _ = cv2.findHomography(src_pt, dst_pt)
            warped = cv2.warpPerspective(img, H, (1280, 720))
            warped_ = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            # print(img.shape)
            # exit(-1)
            # cv2.line(img, (100,900), (902, 900), (0,255,0), 10)
            # cv2.imshow("lines",img)
            # cv2.waitKey(0)
            # cv2.imshow("warped", warped)
            hough_tf(warped, warped_)
            # cv2.imshow("image_warped", warped)
            # cv2.waitKey(0)
    elif sys.argv[2] == 'png':

        files = np.sort(glob.glob(sys.argv[1] + '/*', recursive=True))
        for file in files:
            # print(file)
            img = cv2.imread(file)

            src_pt = np.array([[526, 322], [767, 328],[354, 428],[866, 434] ])
            dst_pt = np.array([[200, img.shape[0]-450], [img.shape[1]-100, img.shape[0]-450], \
                    [200, img.shape[0] - 50], [img.shape[1]-100, img.shape[0] - 50]])

            # cv2.imshow("image", img)
            # cv2.waitKey(0)
            H, _ = cv2.findHomography(src_pt, dst_pt)
            warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
            warped_ = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("image", warped)
            # cv2.waitKey(0)
            # print(img.shape)
            # exit(-1)
            # cv2.line(img, (100,900), (902, 900), (0,255,0), 10)
            # cv2.imshow("lines",img)
            # cv2.waitKey(0)
            # cv2.imshow("warped", warped)
            hough_tf(warped, warped_)
