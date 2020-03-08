import numpy as np 
import cv2

def undistortImage(image):
	K = [[9.037596e+02, 0.000000e+00, 6.957519e+02],[0.000000e+00, 9.019653e+02, 2.242509e+02],\
	[0.000000e+00, 0.000000e+00, 1.000000e+00]]
	K = np.asarray(K)
	D = [-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]
	D = np.asarray(D)
	img = cv2.undistort(image, K, D, None, K)
	img = cv2.GaussianBlur(img,(3,3),0)
	return img 

def curveFit(left,right,img):
	mask = np.zeros_like(img)
	left_lane = np.polyfit(left[1],left[0],2)
	right_lane = np.polyfit(left[1],left[0],2)
	left_poly = np.poly1d(left_lane)
	right_poly = np.poly1d(right_lane)
	wspace = np.linspace(0, img.shape[0]-1, img.shape[0])
	left_fit = left_poly(wspace)
	right_fit = right_poly(wspace)
	coordinates_left = np.hstack([left_fit,wspace])
	coordinates_right = np.hstack([right_fit,wspace])
	points = np.hstack((coordinates_left, coordinates_right))

	cv2.fillPoly(mask, points,(0,255,0))
	image = cv2.warpPerspective(mask, np.linalg.inv(H),(IMG.shape[1], IMG.shape[0]))
	fimage = cv2.addWeighted(IMG, 0.8, image, 0.2, 0.0)
	return fimage





def main():
	fname ='../Problem_2/data_1/data/0000000015.png'
	IMG = cv2.imread(fname)
	print(IMG.shape)
	IMG = IMG[230:IMG.shape[0],:,:]
	IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
	udimg = undistortImage(IMG)
	cv2.imshow("Undistorted",udimg)
	#cv2.imshow("Distorted",IMG)
	cv2.waitKey(0)


if __name__ == '__main__':
	main()