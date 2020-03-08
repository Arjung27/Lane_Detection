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

if __name__ == '__main__':

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
        dst_pt = np.array([[100, 870], [1520, 870], \
                [100, 990], [1520, 990]])

        H, _ = cv2.findHomography(src_pt, dst_pt)
        warped = cv2.warpPerspective(img, H, (1920, 1080))
        cv2.imshow("image_warped", warped)
        cv2.waitKey(0)
