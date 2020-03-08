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

def extractROI(image):
	return img


def main():
	fname ='../Problem_2/data_1/data/0000000015.png'
	IMG = cv2.imread(fname)
	print(IMG.shape)
	IMG = IMG[230:IMG.shape[0],0:980,:]
	IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
	udimg = undistortImage(IMG)
	cv2.imshow("Undistorted",udimg)
	#cv2.imshow("Distorted",IMG)
	cv2.waitKey(0)


if __name__ == '__main__':
	main()