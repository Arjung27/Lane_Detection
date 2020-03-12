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

def predicTurn(img,l_point,r_point):
    i_center = img.shape[1]/2-25
    m_point = l_point + (r_point-l_point)/2
    print(m_point,i_center)
    if abs(m_point-i_center)<15:
        predict = "Straight"
    elif (m_point-i_center)<-30:
        predict = "Left Turn"
    else:
        predict = "Right Turn"
    return predict

def predicTurnRevamped(img,lp_first,lp_second):
    
    if lp_first - lp_second>0:
        predict = "Left Turn"
    elif lp_first - lp_second<0:
        predict = "Right Turn"
    else:
        predict = "Straight"
    return predict

def radiusCurvature(p_x,p_y):
	radC = ((1 + (2*p_y[0]*p_x + p_y[1])**2)**(1.5)) / (2*p_y[0])
	print(np.mean(radC))
	crad = np.mean(radC)
	if crad>3000:
		pred = "Turn Right"
	elif crad<-6000:
		pred = "Turn Left"
	else:
		pred = "Straight"
	return pred


def curveFit(left,right,img):
	mask = np.zeros_like(img).astype(np.uint8)
	left_lane = np.polyfit(left[:,1],left[:,0],2)
	right_lane = np.polyfit(right[:,1],right[:,0],2)
	left_poly = np.poly1d(left_lane)
	right_poly = np.poly1d(right_lane)
	wspace = np.linspace(240,img.shape[0]-1, img.shape[0]-240)
	left_fit = left_poly(wspace)
	right_fit = right_poly(wspace)
	#coordinates_left = np.hstack([left_fit,wspace])
	coordinates_left = np.array([np.transpose(np.vstack([left_fit, wspace]))])
	coordinates_left = coordinates_left[0].astype(np.int32)
	#coordinates_right = np.hstack([right_fit,wspace])
	coordinates_right = np.array([np.transpose(np.vstack([right_fit, wspace]))])
	coordinates_right = coordinates_right[0].astype(np.int32)
	cv2.polylines(img, [coordinates_left], False, (0,0,0),10)
	cv2.polylines(img, [coordinates_right], False, (0,0,0),10)
	lPoint, rPoint = coordinates_left[200][0], coordinates_right[200][0]
	#lPoint, rPoint = coordinates_left[0][0], coordinates_left[150][0]
	#pred = predicTurn(img,lPoint,rPoint)
	pred = radiusCurvature(wspace,right_lane)
	points = np.hstack((coordinates_left, coordinates_right))
	points = np.array([points],dtype=np.int32)[0]
	points = np.array([[[points[0,0],points[0,1]],[points[0,2],points[0,3]],[points[points.shape[0]/2,2]\
		,points[points.shape[0]/2,3]]\
		,[points[points.shape[0]-1,2],points[points.shape[0]-1,3]]\
		,[points[points.shape[0]-1,0],points[points.shape[0]-1,1]],[points[points.shape[0]/2,0],points[points.shape[0]/2,1]]]])
	cv2.fillPoly(mask,points,(0,255,0))
	#image = cv2.warpPerspective(mask, np.linalg.inv(H),(IMG.shape[1], IMG.shape[0]))
	fimage = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)

	return fimage, pred

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