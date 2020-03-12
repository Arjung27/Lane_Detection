import os
import sys
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from detection_pipeline import curveFit


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def hist_equalize(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    N = img.shape[0]*img.shape[1]
    cdf_m = cdf_m*255/N
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img = cdf[img]

    return img

def hough_tf(img, img_):

    lower_color_bound = (0, 150, 180)
    upper_color_bound = (200, 180, 190) 

    img_thresh = cv2.inRange(img, lower_color_bound, upper_color_bound) ## Does a yellow filter and keeps the white lanes as well. 


    edges = cv2.Canny(img_thresh, 20, 20, apertureSize=3) ## finds lines on a thresholded image using canny. output is edges in random directions

    edges = cv2.GaussianBlur(edges,(35,35),0) # Blurred the random direction edges to no longer see them as edges but as spots on lanes 

    

    if np.median(img_) < 100: # when entering the shaded area, enhance the edges and then do the operation on the image

        img_sharped = cv2.addWeighted(img, 8, cv2.blur(img, (35, 35)), -8, 128) # sharpen

        lower_color_bound = (140, 150, 135) ## use a filter to find points on the lanes
        upper_color_bound = (200, 250, 220) ## use a filter to find points on the lanes , this was very tricky and a narrow range. 
        img_thresh = cv2.inRange(img_sharped, lower_color_bound, upper_color_bound) # thresholding 
        
        edges = cv2.Canny(img_thresh, 20, 20, apertureSize=3)

        edges = cv2.GaussianBlur(edges,(35,35),0)
    
    elif np.median(img) > 158: # when exiting the shaded area, 

        lower_color_bound = (120, 180, 200) ## use a filter to find points on the lanes
        upper_color_bound = (200, 250, 250) ## use a filter to find points on the lanes , this was very tricky and a narrow range. 
        img_thresh = cv2.inRange(img, lower_color_bound, upper_color_bound) # thresholding 
        
        edges = cv2.Canny(img_thresh, 20, 20, apertureSize=3)
        edges = cv2.GaussianBlur(edges,(35,35),0)


    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 10, 10)

    left_pts = []
    right_pts = []
    for x1, y1, x2, y2, in lines[:,0,:]:

        if np.abs((y2 - y1)/(x2 - x1)) < 5: ## allow lines which are only > ~65degs 
            continue

        if (x1 < 120 or x2 < 130) or (x1 > 300 or x2 > 350) and (x1 < img.shape[1]-300 or x2 < img.shape[1]-350):
            continue

        if x1 < img.shape[1]/2:

            left_pts.append(([x1,y1]))
        elif x1 > img.shape[1]/2:

            right_pts.append(([x1,y1]))     
        if x2 < img.shape[1]/2:

            left_pts.append(([x1,y1]))
        elif x2 > img.shape[1]/2:
            right_pts.append(([x1,y1]))
                         
        cv2.line(img, (x1,y1), (x2, y2), (0,255,0), 2)
    left_pts = np.asarray(left_pts)
    right_pts = np.asarray(right_pts)
    if left_pts.shape == (0,):
        if right_pts.shape == (0,):
            left_pts = [[0,0], [0,0]]
            right_pts = [[0,0], [0,0]]
        else:    
            left_pts = np.zeros_like(right_pts)
    elif right_pts.shape == (0,):
        if left_pts.shape == (0,):
            left_pts = [[0,0], [0,0]]
            right_pts = [[0,0], [0,0]]
        else:    
            right_pts = np.zeros_like(left_pts)
        
    cv2.imshow(" s", img)
    #cv2.waitKey(0)
    return left_pts, right_pts, img



def hough_tf_image(img, img_):

    _, img_thresh = cv2.threshold(img_, 220, 255, cv2.THRESH_BINARY) # thresholding image to get a high intensity


    edges = cv2.GaussianBlur(img_thresh,(49,49),0) # Blur the edges to so that random direction lines are just seen as edge patches rather than edges


    if np.median(img_) < 100:

        _, img_thresh2 = cv2.threshold(img_, 80, 255, cv2.THRESH_BINARY) # threshold the histogram equalised img

        edges2 = cv2.Canny(img_thresh2, 3, 3, apertureSize=3) # 

        edges2 = cv2.GaussianBlur(edges2,(5,5),0)

        edges = edges + edges2
        edges = adjust_gamma(edges, 1.4)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, 10, 10)
    left_pts = []
    right_pts = []
    for x1, y1, x2, y2, in lines[:,0,:]:
        if np.abs((y2 - y1)/(x2 - x1)) < 2:
            continue

        if (x1 < 20 or x2 < 30) or (x1 > img.shape[1]-30 or x2 > img.shape[1]-35):
            continue
        
        if x1 < img.shape[1]/2:
            left_pts.append(([x1,y1]))
        elif x1 > img.shape[1]/2:
            right_pts.append(([x1,y1]))     
        if x2 < img.shape[1]/2:
            left_pts.append(([x1,y1]))
        elif x2 > img.shape[1]/2:
            right_pts.append(([x1,y1]))
                         
        cv2.line(img, (x1,y1), (x2, y2), (0,255,0), 2)
    left_pts = np.asarray(left_pts)
    right_pts = np.asarray(right_pts)
    if left_pts.shape == (0,):
        left_pts = np.zeros_like(right_pts)
    elif right_pts.shape == (0,):
        right_pts = np.zeros_like(left_pts)
    cv2.imshow(" s", img)
    return left_pts, right_pts, img 

def hist_count(img, img_):
    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    graph = np.zeros([img.shape[0], img.shape[1], 2])
    graph[:,:,0] = img[:,:,0]
    graph[:,:,1] = (img[:,:,1] + img[:,:,2])/(abs(img[:,:,1] - img[:,:,2]) + 1)
    X = np.arange(0,graph.shape[0])
    Y = np.arange(0,graph.shape[1])

    X, Y = np.meshgrid(X, Y)
    Z = (img[X, Y,1] + img[X,Y,2])/(abs(img[X,Y,1] - img[X,Y,2]) + 1)
    surf = ax1.plot_surface(X, Y, Z)
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    Z2 = img[X,Y,0]
    surf = ax2.plot_surface(X, Y, Z2)
    plt.show()
    exit(-1)

    cv2.imshow("lines",img)
    #cv2.waitKey(0)




if __name__ == '__main__':

    if sys.argv[2] == 'mp4':
        video_file = sys.argv[1]
        cap = cv2.VideoCapture(video_file)
        vidWriter = cv2.VideoWriter("./video_output.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 24, (1280,720))

        arrl_store = []
        arrr_store = []
        while (cap.isOpened()):

            ret, img = cap.read()
            if ret == False:
                break

            src_pt = np.array([[407, 615], [886, 615],[312, 677],[962, 677] ])
            dst_pt = np.array([[150, 630], [1080, 630], \
                    [150, 701], [1080, 701]])

            H, _ = cv2.findHomography(src_pt, dst_pt)
            warped = cv2.warpPerspective(img, H, (1280, 720))
            warped_ = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            left_pts, right_pts, lines = hough_tf(warped, warped_)
            # hist_count(warped, warped_)

            left_pts = np.asarray(left_pts)
            right_pts = np.asarray(right_pts)

            if len(arrl_store)<3:
                arrl_store.append(left_pts)
                arrr_store.append(right_pts)
            elif right_pts[right_pts.shape[0]-2,0]!=0 and left_pts[left_pts.shape[0]-2,0]!=0:
                arrr_store.pop()
                arrr_store.append(right_pts)
                arrl_store.pop()
                arrl_store.append(left_pts)
            arr_templ = np.asarray(arrl_store)
            arr_tempr = np.asarray(arrr_store)
            lefty = arr_templ[arr_templ.shape[0]-1]
            righty = arr_tempr[arr_tempr.shape[0]-1]  
            flag =1  
            lines,prediction = curveFit(lefty,righty,lines,flag)
            Hinv = np.linalg.inv(H)
            Hinv = Hinv/Hinv[2,2]
            blank_white = 255*np.ones([720, 1280, 3])
            blank_backProj = cv2.warpPerspective(blank_white, Hinv, (1280, 720))
            _, thresh1 = cv2.threshold(blank_backProj, 127, 255, cv2.THRESH_BINARY_INV)

            patch = np.logical_and(img, thresh1)
            img_dummy = np.zeros_like(img)
            img_dummy[patch] = img[patch]
            backProjected = cv2.warpPerspective(lines, Hinv, (1280, 720))

            img_dummy = img_dummy + backProjected

            cv2.putText(img_dummy, prediction, (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2,\
            cv2.LINE_AA)
            cv2.imshow("test", img_dummy)

            vidWriter.write(img_dummy.astype(np.uint8))
    
    elif sys.argv[2] == 'png':

        files = np.sort(glob.glob(sys.argv[1] + '/*', recursive=True))
        vidWriter = cv2.VideoWriter("video_output.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 24, (1392,512))
        arrl_store = []
        arrr_store = []
        for file in files:
            img = cv2.imread(file)

            src_pt = np.array([[526, 322], [767, 328],[354, 428],[866, 434] ])
            dst_pt = np.array([[200, img.shape[0]-450], [img.shape[1]-100, img.shape[0]-450], \
                    [200, img.shape[0] - 50], [img.shape[1]-100, img.shape[0] - 50]])

            H, _ = cv2.findHomography(src_pt, dst_pt)
            warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
            warped_ = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            left_pts, right_pts, lines = hough_tf_image(warped, warped_)
            if len(arrl_store)<3:
                arrl_store.append(left_pts)
                arrr_store.append(right_pts)
            elif right_pts[right_pts.shape[0]-2,0]!=0 and left_pts[left_pts.shape[0]-2,0]!=0:
                arrr_store.pop()
                arrr_store.append(right_pts)
                arrl_store.pop()
                arrl_store.append(left_pts)
            arr_templ = np.asarray(arrl_store)
            arr_tempr = np.asarray(arrr_store)
            lefty = arr_templ[arr_templ.shape[0]-1]
            righty = arr_tempr[arr_tempr.shape[0]-1]
            flag = 0    
            lines,prediction = curveFit(lefty,righty,lines,flag)
            Hinv = np.linalg.inv(H)
            Hinv = Hinv/Hinv[2,2]
            blank_white = 255*np.ones([img.shape[0], img.shape[1], 3])
            blank_backProj = cv2.warpPerspective(blank_white, Hinv, (img.shape[1], img.shape[0]))
            _, thresh1 = cv2.threshold(blank_backProj, 127, 255, cv2.THRESH_BINARY_INV)
            patch = np.logical_and(img, thresh1)
            img_dummy = np.zeros_like(img)
            img_dummy[patch] = img[patch]
            backProjected = cv2.warpPerspective(lines, Hinv, (img.shape[1], img.shape[0]))
            img_dummy = img_dummy + backProjected
            cv2.putText(img_dummy, prediction, (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2,\
            cv2.LINE_AA)

            vidWriter.write(img_dummy.astype(np.uint8))

    
    vidWriter.release()
