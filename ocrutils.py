import cv2
import numpy as np
import imutils
from sklearn.metrics import euclidean_distances

DOC_WIDTH, DOC_HEIGHT = 1098, 648

def pil_as_array(pil_img):
    rgb_img = pil_img.convert("RGB")
    open_cv_image = np.array(rgb_img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image

def gamma_correction(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

def get_padding(img):
    '''add padding equivalent to 10% of the document size and return image'''
    image = imutils.resize(img, height = 500)
    
    top = int(0.1 * DOC_HEIGHT)
    bottom = top
    left = int(0.1 * DOC_WIDTH)
    right = left
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.imshow("image with padding", image)
    cv2.waitKey(0)
    
    return image

def get_edges(img):
    '''input an image opened by invoking cv2.imread.
        output the same image reduced to its edges.'''
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(gammaCorrection(img, 1.05), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 70, 150)
    
    return edged

def get_all_boxes(edged):
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
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
    screenCnt[:, 0] -= int(0.1 * DOC_WIDTH)
    screenCnt[:, 1] -= int(0.1 * DOC_HEIGHT)
    close_pairs = np.array(np.where(~((euclidean_distances(screenCnt, screenCnt) > 50) | (euclidean_distances(screenCnt, screenCnt) == 0)))).T

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
    result = screenCnt[mask]

    #sorted_points = np.array(sorted(result, key=lambda x: x[0]))
    left, right, top, bottom = min(result[:, 0]), max(result[:, 0]), min(result[:, 1]), max(result[:, 1])

    topleft = min(result, key=lambda x: euclidean_distances([x], [(left, top)]))
    topright = min(result, key=lambda x: euclidean_distances([x], [(right, top)]))
    bottomleft = min(result, key=lambda x: euclidean_distances([x], [(left, bottom)]))
    bottomright = min(result, key=lambda x: euclidean_distances([x], [(right, bottom)]))
    
    return topleft, topright, bottomleft, bottomright

def transform_image(img):
    '''the official driver code for the whole process.
    Accepts an image as input, for OCRImg class
    Outputs the image required for pyt.image_to_string() call to OCR Engine.
    Assume that the image has already be resized'''
    
    #read the image and get its edges
    edged = get_edges(img)    
    boxes = get_all_boxes(edged)
    topleft, topright, bottomleft, bottomright = select_coords(boxes) #something sus here

    #initialize points: input convert to top-down output
    input_pts = np.float32([list(coord) for coord in [topleft, bottomleft, bottomright, topright]])
    output_pts = np.float32([[0, 0],
                            [0, DOC_HEIGHT - 1],
                            [DOC_WIDTH - 1, DOC_HEIGHT - 1],
                            [DOC_WIDTH - 1, 0]])
    
    # Compute the perspective transform mat
    transform_mat = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(img, transform_mat, (DOC_WIDTH, DOC_HEIGHT), flags=cv2.INTER_LINEAR)
    
    return out