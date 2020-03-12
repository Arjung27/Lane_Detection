import os
import sys
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from detection_pipeline import curveFit
from detection_pipeline import predicTurn,predicTurnRevamped


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
    
    # print(np.median(img))
    if np.median(img_) < 100: # when entering the shaded area, enhance the edges and then do the operation on the image

        img_sharped = cv2.addWeighted(img, 8, cv2.blur(img, (35, 35)), -8, 128) # sharpen
        # cv2.imshow("median", img_sharped)
        # cv2.waitKey(0)
        lower_color_bound = (140, 150, 135) ## use a filter to find points on the lanes
        upper_color_bound = (200, 250, 220) ## use a filter to find points on the lanes , this was very tricky and a narrow range. 
        img_thresh = cv2.inRange(img_sharped, lower_color_bound, upper_color_bound) # thresholding 
        
        edges = cv2.Canny(img_thresh, 20, 20, apertureSize=3)
        # cv2.imshow("a", edges)
        # cv2.waitKey(0)
        edges = cv2.GaussianBlur(edges,(35,35),0)
    
    elif np.median(img) > 158: # when exiting the shaded area, 

        # img_sharped = cv2.addWeighted(img, 4, cv2.blur(img, (35, 35)), -4, 128) # sharpen
        # cv2.imshow("median", img_sharped)
        # cv2.waitKey(0)
        lower_color_bound = (120, 180, 200) ## use a filter to find points on the lanes
        upper_color_bound = (200, 250, 250) ## use a filter to find points on the lanes , this was very tricky and a narrow range. 
        img_thresh = cv2.inRange(img, lower_color_bound, upper_color_bound) # thresholding 
        
        edges = cv2.Canny(img_thresh, 20, 20, apertureSize=3)
        # cv2.imshow("a", edges)
        # cv2.waitKey(0)
        edges = cv2.GaussianBlur(edges,(35,35),0)

        # cv2.imshow("a", edges)
        # cv2.waitKey(0)


    # cv2.imshow(" s", edges)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 10, 10)
    #left_pts = np.array([])
    #right_pts = np.array([])
    left_pts = []
    right_pts = []
    for x1, y1, x2, y2, in lines[:,0,:]:

        if np.abs((y2 - y1)/(x2 - x1)) < 5: ## allow lines which are only > ~65degs 
            continue

        if (x1 < 120 or x2 < 130) or (x1 > 300 or x2 > 350) and (x1 < img.shape[1]-300 or x2 < img.shape[1]-350):
            continue

        if x1 < img.shape[1]/2:
            #left_pts = np.append(left_pts, [x1, y1])
            left_pts.append(([x1,y1]))
        elif x1 > img.shape[1]/2:
            #right_pts = np.append(right_pts, [x1, y1])
            right_pts.append(([x1,y1]))     
        if x2 < img.shape[1]/2:
            #left_pts = np.append(left_pts, [x2, y2])
            left_pts.append(([x1,y1]))
        elif x2 > img.shape[1]/2:
            #right_pts = np.append(right_pts, [x2, y2])
            right_pts.append(([x1,y1]))
                         
        cv2.line(img, (x1,y1), (x2, y2), (0,255,0), 2)
    left_pts = np.asarray(left_pts)
    right_pts = np.asarray(right_pts)
    if left_pts.shape == (0,):
        left_pts = np.zeros_like(right_pts)
    elif right_pts.shape == (0,):
        right_pts = np.zeros_like(left_pts)

    #print(left_pts.shape)
    #print(right_pts.shape)

    return left_pts, right_pts, img



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
        # img_sharped = hist_equalize(img_) # Equalise the image when image is dull
        # cv2.imshow("edges1", img_sharped)
        # cv2.waitKey(0)
        _, img_thresh2 = cv2.threshold(img_, 80, 255, cv2.THRESH_BINARY) # threshold the histogram equalised img
        # cv2.imshow("edges1", img_thresh2)
        # cv2.waitKey(0)
        edges2 = cv2.Canny(img_thresh2, 3, 3, apertureSize=3) # 
        # edges2 = cv2.addWeighted(edges2, 4, cv2.blur(edges2, (35, 35)), -4, 128)
        edges2 = cv2.GaussianBlur(edges2,(5,5),0)
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
    cv2.imshow(" s", edges)
    cv2.waitKey(0)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, 10, 10)
    for x1, y1, x2, y2, in lines[:,0,:]:
        if np.abs((y2 - y1)/(x2 - x1)) < 2:
            continue

        if (x1 < 20 or x2 < 30) or (x1 > img.shape[1]-30 or x2 > img.shape[1]-35):
            continue
        # if (x1 < img.shape[1]/2 and x2 < img.shape[1]/2)
        #     x_left.append(x1)
        #     x_left.
        cv2.line(img, (x1,y1), (x2, y2), (0,255,0), 2)
    cv2.imshow("lines",img)
    cv2.waitKey(0)




if __name__ == '__main__':

    if sys.argv[2] == 'mp4':
        video_file = sys.argv[1]
        cap = cv2.VideoCapture(video_file)

        arrl_store = []
        arrr_store = []
        while (cap.isOpened()):

            ret, img = cap.read()
            if ret == False:
                break

            src_pt = np.array([[407, 615], [886, 615],[312, 677],[962, 677] ])
            dst_pt = np.array([[150, 630], [1080, 630], \
                    [150, 700], [1080, 700]])

            H, _ = cv2.findHomography(src_pt, dst_pt)
            warped = cv2.warpPerspective(img, H, (1280, 720))
            warped_ = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            left_pts, right_pts, lines = hough_tf(warped, warped_)
            if len(arrl_store)<3:
                arrl_store.append(left_pts)
                arrr_store.append(right_pts)
            elif right_pts[right_pts.shape[0]-5,0]!=0 and left_pts[left_pts.shape[0]-5,0]!=0:
                arrr_store.pop()
                arrr_store.append(right_pts)
                arrl_store.pop()
                arrl_store.append(left_pts)
            arr_templ = np.asarray(arrl_store)
            arr_tempr = np.asarray(arrr_store)
            lefty = arr_templ[arr_templ.shape[0]-1]
            righty = arr_tempr[arr_tempr.shape[0]-1]    
            lines,prediction = curveFit(lefty,righty,lines)
            #print(prediction)
            Hinv = np.linalg.inv(H)
            Hinv = Hinv/Hinv[2,2]
            blank_white = 255*np.ones([720, 1280, 3])
            # blank_black = 255*np.zeros([720, 1280, 3])
            blank_backProj = cv2.warpPerspective(blank_white, Hinv, (1280, 720))
            _, thresh1 = cv2.threshold(blank_backProj, 127, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow("test", thresh1)
            # cv2.waitKey(0)
            patch = np.logical_and(img, thresh1)
            img_dummy = np.zeros_like(img)
            img_dummy[patch] = img[patch]
            backProjected = cv2.warpPerspective(lines, Hinv, (1280, 720))
            # patch2 = np.logical_or(img_dummy, backProjected)
            img_dummy = img_dummy + backProjected
            #prediction = predicTurn(img_dummy,L,R)
            cv2.putText(img_dummy, prediction, (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2,\
            cv2.LINE_AA)
            # print(backProjected)
            cv2.imshow("test", img_dummy)
            cv2.waitKey(0)

    elif sys.argv[2] == 'png':

        files = np.sort(glob.glob(sys.argv[1] + '/*', recursive=True))
        for file in files:
            # print(file)
            img = cv2.imread(file)

            src_pt = np.array([[526, 322], [767, 328],[354, 428],[866, 434] ])
            dst_pt = np.array([[200, img.shape[0]-450], [img.shape[1]-100, img.shape[0]-450], \
                    [200, img.shape[0] - 50], [img.shape[1]-100, img.shape[0] - 50]])

            H, _ = cv2.findHomography(src_pt, dst_pt)
            warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
            warped_ = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            lines = hough_tf_image(warped, warped_)


