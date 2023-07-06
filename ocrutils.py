import cv2
import numpy as np
from imutils import resize, grab_contours
from sklearn.metrics import euclidean_distances

DOC_WIDTH, DOC_HEIGHT = 1098, 648

def pil_as_array(pil_img):
    rgb_img = pil_img.convert("RGB")
    open_cv_image = np.array(rgb_img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image

def grayscale(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    return gray

def morph(src):
    return cv2.morphologyEx(src, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))

def remove_noise(src):
    return cv2.fastNlMeansDenoising(src, None, 10, 10, 7)

def get_edges(src):
    '''input an image opened by invoking cv2.imread.
        output the same image reduced to its edges.'''
    # convert the image to grayscale, blur it, and find edges
    # in the image
    edged = cv2.Canny(src, 30, 30)
    
    return edged

def get_all_boxes(edged):
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

def select_coords(screenCnt):
    screenCnt = screenCnt.reshape(len(screenCnt), 2)
    close_pairs = np.array(np.where(~((euclidean_distances(screenCnt, screenCnt) > 10) | (euclidean_distances(screenCnt, screenCnt) == 0)))).T

    if len(close_pairs):
        pair_cleaned = []

        for p1, p2 in close_pairs:
            a = False
            for x1, x2 in pair_cleaned:
                if p1 == x2:
                    a = not a
                    break
            if not a:
                pair_cleaned.append((p1, p2))

        remove_points = np.array(pair_cleaned)[:, 1]

        mask = np.ones(len(screenCnt), dtype=bool) 
        mask[remove_points] = False
        screenCnt = screenCnt[mask]

    #sorted_points = np.array(sorted(screenCnt, key=lambda x: x[0]))
    left, right, top, bottom = min(screenCnt[:, 0]), max(screenCnt[:, 0]), min(screenCnt[:, 1]), max(screenCnt[:, 1])

    topleft = min(screenCnt, key=lambda x: euclidean_distances([x], [(left, top)]))
    topright = min(screenCnt, key=lambda x: euclidean_distances([x], [(right, top)]))
    bottomleft = min(screenCnt, key=lambda x: euclidean_distances([x], [(left, bottom)]))
    bottomright = min(screenCnt, key=lambda x: euclidean_distances([x], [(right, bottom)]))
    
    return topleft, topright, bottomleft, bottomright

def crop_image(img, topleft, topright, bottomleft, bottomright):
    max_h, max_w = img.shape[:2]

    #initialize points: input convert to top-down output
    input_pts = np.float32([list(coord) for coord in [topleft, bottomleft, bottomright, topright]])
    output_pts = np.float32([[0, 0],
                            [0, max_h - 1],
                            [max_w - 1, max_h - 1],
                            [max_w - 1, 0]])
    
    #plt.imshow(cv2.drawContours(img, [np.array([topleft, topright, bottomright, bottomleft]).reshape((4, 1, 2))], -1, (255,0,0), 2))
    #cv2.imshow("edged", edged)
    #cv2.waitKey(0)
    
    # Compute the perspective transform mat
    transform_mat = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(img, transform_mat, (max_w, max_h), flags=cv2.INTER_LINEAR)
    return out

def transform_image(img):
    '''the official driver code for the whole process.
    Accepts an image as input, for OCRImg class
    Outputs the image required for pyt.image_to_string() call to OCR Engine.
    Assume that the image has already be resized'''
    
    #read the image and get its edges
    img_resize = resize(img, height = 500)
    img_gray = grayscale(img_resize)
    img_morph = morph(img_gray)
    reduced_noise = remove_noise(img_morph)
    edged = get_edges(reduced_noise)
    topleft, topright, bottomleft, bottomright = select_coords(get_all_boxes(edged))
    transformed = crop_image(img_resize, topleft, topright, bottomleft, bottomright)
    
    return transformed