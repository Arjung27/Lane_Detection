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
    _, img_thresh2 = cv2.threshold(img, 10, 255, cv2.THRESH_TOZERO)
    # hist,bins = np.histogram(img_thresh2.flatten(),256,[0,256])
    # cdf = hist.cumsum()
    # cdf_m = np.ma.masked_equal(cdf,0)
    # N = img.shape[0]*img.shape[1]
    # cdf_m = cdf_m*255/N
    # cdf = np.ma.filled(cdf_m,0).astype('uint8')
    # img2 = cdf[img_thresh2]
    plt.hist(img.ravel(), bins=256, range=(0.0, 256.0))
    plt.show()
    hist_b = cv2.equalizeHist(img_thresh2[:,:,0])
    hist_g = cv2.equalizeHist(img_thresh2[:,:,1])
    hist_r = cv2.equalizeHist(img_thresh2[:,:,2])
    img2 = np.dstack([hist_b, hist_g, hist_r])
    plt.hist(img2.ravel(), bins=256, range=(10.0, 256.0))
    plt.show()
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
        hist_equalize(img)
        # src_pt = np.array([[549.177, 1025.69], [1892.4, 1025.69],\
        #                     [564.661, 1010.21], [1876.92, 1010.21]])
        # dst_pt = np.array([[100, 0], [img.shape[1]-100, 0], \
        #         [100, img.shape[0]-100], [img.shape[1]-100, img.shape[0]-100]])

        # H, _ = cv2.findHomography(src_pt, dst_pt)
        # warped = cv2.warpPerspective(img, H, (1920, 1080))
        # cv2.imshow("image_warped", warped)
        # cv2.waitKey(0)