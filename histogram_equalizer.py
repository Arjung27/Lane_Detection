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
    img = cdf[img]
    # cv2.imshow("image", img2)
    # cv2.waitKey(0)
    # plt.imshow(img2)
    # plt.show()
    return img

def hough_tf(img, img_):

    # img_ = adjust_gamma(img_, 2)
    lower_color_bound = (0, 150, 180)
    upper_color_bound = (200, 180, 190) 
    # _, img_ = cv2.threshold(img_, 200, 255, cv2.THRESH_BINARY)
    img_thresh = cv2.inRange(img, lower_color_bound, upper_color_bound) ## Does a yellow filter and keeps the white lanes as well. 
    # cv2.imshow("a", img_)
    # cv2.waitKey(0)

    edges = cv2.Canny(img_thresh, 20, 20, apertureSize=3) ## finds lines on a thresholded image using canny. output is edges in random directions
    # cv2.imshow("a", edges)
    # cv2.waitKey(0)
    edges = cv2.GaussianBlur(edges,(35,35),0) # Blurred the random direction edges to no longer see them as edges but as spots on lanes 
    # cv2.imshow("a", edges)
    # cv2.waitKey(0)
    
    if np.median(img_) < 100: # when entering the shaded area, enhance the edges and then do the operation on the image

        img_sharped = cv2.addWeighted(img, 2, cv2.blur(img, (35, 35)), -2, 128) # sharpen
        # cv2.imshow("median", img_sharped)
        # cv2.waitKey(0)
        lower_color_bound = (140, 150, 135) ## use a filter to find points on the lanes
        upper_color_bound = (200, 250, 220) ## use a filter to find points on the lanes , this was very tricky and a narrow range. 
        # _, img_ = cv2.threshold(img_, 200, 255, cv2.THRESH_BINARY)
        img_thresh = cv2.inRange(img_sharped, lower_color_bound, upper_color_bound) # thresholding 
        

        edges = cv2.Canny(img_thresh, 20, 20, apertureSize=3)
        edges = cv2.GaussianBlur(edges,(35,35),0)

        cv2.imshow("a", edges)
        cv2.waitKey(0)


    # cv2.imshow(" s", edges)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 10, 10)

    for x1, y1, x2, y2, in lines[:,0,:]:

        if np.abs((y2 - y1)/(x2 - x1)) < 2: ## allow lines which are only > ~65degs 
            continue
        cv2.line(img, (x1,y1), (x2, y2), (0,255,0), 2)
    cv2.imshow("lines",img)
    cv2.waitKey(0)

def hough_tf_image(img, img_):

    # img_ = hist_equalize(img_)
    # img_ = adjust_gamma(img_, 2)
    # img_ = cv2.GaussianBlur(img_,(5,5),0)
    _, img_thresh = cv2.threshold(img_, 220, 255, cv2.THRESH_BINARY) # thresholding image to get a high intensity
    # cv2.imshow("edges1", img_thresh)
    # cv2.waitKey(0)

    # REMOVED CANNY FOR NON SHADDY REGIONS TO GET MORE HOUGH LINES
    # edges = cv2.Canny(img_thresh, 20, 20, apertureSize=3) # find edges on thresholded images
    # cv2.imshow("edges1", edges)
    # cv2.waitKey(0)
    edges = cv2.GaussianBlur(img_thresh,(49,49),0) # Blur the edges to so that random direction lines are just seen as edge patches rather than edges
    # cv2.imshow("edges1", edges)
    # cv2.waitKey(0)

    if np.median(img_) < 100:
        # img_sharped = cv2.addWeighted(img_, 4, cv2.blur(img_, (35, 35)), -4, 128)
        # cv2.imshow("edges1", img_)
        # cv2.waitKey(0)
        img_sharped = hist_equalize(img_) # Equalise the image when image is dull
        # cv2.imshow("edges1", img_sharped)
        # cv2.waitKey(0)
        _, img_thresh2 = cv2.threshold(img_sharped, 180, 255, cv2.THRESH_BINARY) # threshold the histogram equalised img
        # cv2.imshow("edges1", img_thresh2)
        # cv2.waitKey(0)
        edges2 = cv2.Canny(img_thresh2, 3, 3, apertureSize=3) # 
        # edges2 = cv2.addWeighted(edges2, 4, cv2.blur(edges2, (35, 35)), -4, 128)
        # edges2 = cv2.GaussianBlur(edges2,(49,49),0)
        # cv2.imshow("edges1", edges2)
        # cv2.waitKey(0)
        # cv2.imshow("edges2", edges2)
        # cv2.waitKey(0)
        edges = edges + edges2
        edges = adjust_gamma(edges, 1.4)
        # cv2.imshow("edges1", edges)
        # cv2.waitKey(0)
        # print(np.max(edges))
    # edges = adjust_gamma(edges, 2)
    # cv2.imshow(" s", img_thresh)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, 10, 10)

    for x1, y1, x2, y2, in lines[:,0,:]:
        if np.abs((y2 - y1)/(x2 - x1)) < 2:
            continue
        cv2.line(img_, (x1,y1), (x2, y2), (0,255,0), 2)
    cv2.imshow("lines",img_)
    cv2.waitKey(0)   

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

            hough_tf(warped, warped_)

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
            hough_tf_image(warped, warped_)
