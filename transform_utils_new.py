import cv2
import numpy as np
from imutils import grab_contours
from sklearn.metrics import euclidean_distances

def blur(img, kernel_size=7):
    return cv2.medianBlur(img, kernel_size)

def edge(img, threshold=50):
    aspect_ratio = img.shape[1] / img.shape[0]
    threshold_height = int(threshold / aspect_ratio)
    return cv2.Canny(img, threshold, threshold_height, apertureSize=7)

def get_largest_contour(edged: np.ndarray):
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) >= 4:
			screenCnt = approx
			break

	return screenCnt

def get_perspective_corners(screenCnt: np.ndarray):
    screenCnt = screenCnt.reshape(len(screenCnt), 2)
    left, right, top, bottom = min(screenCnt[:, 0]), max(screenCnt[:, 0]), min(screenCnt[:, 1]), max(screenCnt[:, 1])

    topleft = min(screenCnt, key=lambda x: euclidean_distances([x], [(left, top)]))
    topright = min(screenCnt, key=lambda x: euclidean_distances([x], [(right, top)]))
    bottomleft = min(screenCnt, key=lambda x: euclidean_distances([x], [(left, bottom)]))
    bottomright = min(screenCnt, key=lambda x: euclidean_distances([x], [(right, bottom)]))
    
    return topleft, topright, bottomleft, bottomright

def transform_image(img, topleft, topright, bottomleft, bottomright):
    max_h, max_w = img.shape[:2]

    #initialize points: input convert to top-down output
    input_pts = np.float32([list(coord) for coord in [topleft, bottomleft, bottomright, topright]])
    output_pts = np.float32([[0, 0],
                            [0, max_h - 1],
                            [max_w - 1, max_h - 1],
                            [max_w - 1, 0]])
    
    # Compute the perspective transform mat
    transform_mat = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(img, transform_mat, (max_w, max_h), flags=cv2.INTER_LINEAR)
    return out

def remove_glare(img: np.ndarray, brightness: int):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_lab[:, :, 0] = brightness
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

def extract_card(img: np.ndarray):
    img_rem_glare = remove_glare(img, 100)
    edges = edge(img_rem_glare)
    largest_contour = get_largest_contour(edges)
    return get_perspective_corners(largest_contour)

def transform_card(img: np.ndarray):
    return transform_image(img, *extract_card(img))
