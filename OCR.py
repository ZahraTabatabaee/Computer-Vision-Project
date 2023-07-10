import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import math


def reorder(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p1_index,p0_index, p3_index,p2_index]]
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle/np.pi * 180 > 45:
            # 这个点为p2 - this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p1_index,p0_index, p3_index,p2_index]]
        else:
            # 这个点为p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p1_index,p0_index, p3_index,p2_index]]

def Reflect101(img,filter_size):
  '''
    Do not use loop (like while and for)
    Do not use libraries
    calculate averaging filter
    input(s):
      img (ndarray): input image
      filter_size (ndarray): filter size
    output(s):
      image (ndarray): computed Reflect101
  '''
  image = np.pad(img, pad_width=((filter_size//2, filter_size//2), (filter_size//2, filter_size//2)), mode='reflect', reflect_type='odd')

  return image

def Averaging_Blurring(img, filter_size):
  '''
    Do not use libraries
    input(s):
      img (ndarray): input image
      filter_size (ndarray): filter size
    output(s):
      result (ndarray): computed averaging blurring
  '''
  image = Reflect101(img, filter_size)
  result = np.zeros((img.shape))
  avg_filter = np.ones((filter_size , filter_size))/ (filter_size * filter_size)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      result[i,j] = np.convolve(image[i :i + filter_size , j : j + filter_size].flatten(), avg_filter.flatten() , mode='valid')
  # result[:,:] = np.convolve(image[filter_size//2 : - filter_size //2 , filter_size //2 : -filter_size //2] , avg_filter , mode='valid')

  return result

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def scan(img , kernel_size ,t_lower , t_higher , output_name):
  rectangle = False
  # converting to grayscale
  gray_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # apply gaussian blur to remove noise
  blurred_im = cv2.GaussianBlur(gray_im, kernel_size, 0)
  blurred_im = Averaging_Blurring(gray_im , kernel_size[0])
  if blurred_im.dtype != np.uint8:
    blurred_im = blurred_im.astype(np.uint8)
  edges_im_blurred = cv2.Canny(blurred_im, t_lower, t_higher)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
  dilated_im = cv2.dilate(edges_im_blurred, kernel, iterations=2)

  contours, hierarchy = cv2.findContours(dilated_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  card_contour = max(contours, key=cv2.contourArea)
  # peri = cv2.arcLength(card_contour, True)
  # card_contour = cv2.approxPolyDP(card_contour, 0.03 * peri, True)
  img_copy = img.copy()
  cv2.drawContours(img_copy, [card_contour], -1, (0, 0, 255), 2)
  hull = cv2.convexHull(card_contour)
  epsilon = 0.05 * cv2.arcLength(card_contour, True)
  approx = cv2.approxPolyDP(card_contour, epsilon, True)
  if len(approx) != 4:
    approx = cv2.approxPolyDP(hull, epsilon, True)
    box = reorder(approx.reshape(4, 2)).astype(np.float32)
    # rect = cv2.minAreaRect(hull)
    # box = cv2.boxPoints(rect)
    # box = reorder(box)
    print("not a rectangle")
    # box = np.int0(box)
  else:
    rectangle = True
    box = reorder(approx.reshape(4, 2)).astype(np.float32)

  # Reshape the points to a 3D array
  points = box.reshape((-1, 1, 2))
  points = np.int0(points)
  # Draw the rectangle on the image
  cv2.polylines(img_copy, [points], isClosed=True, color=(0, 255, 0), thickness=2)
#   plt.imshow(img_copy , cmap = 'gray')


  pts1 = np.float32(box)
  print(pts1)
  if calculate_distance(box[0], box[1]) < calculate_distance(box[0], box[3]) :
    print("yes")
    pts2 = np.float32([[860, 0], [860 , 540], [0, 540] , [0, 0]])
  else :
    pts2 = np.float32([[0, 0], [860 , 0], [860, 540] , [0, 540]])


  if rectangle == False :
      if calculate_distance(box[0], box[1]) < calculate_distance(box[0], box[3]) :
        print("yes")
        pts2 = np.float32([[0, 0], [0 , 540], [860, 540] , [860, 0]])
      else :
        pts2 = np.float32([[860, 0], [0 , 0], [0, 540] , [860, 540]])

  M = cv2.getPerspectiveTransform(pts1, pts2)
  img_copy = img.copy()
  dst = cv2.warpPerspective(img_copy, M, (860, 540))
#   plt.imshow(dst , cmap = 'gray')
  cv2.imwrite(output_name, dst)

img7 = cv2.imread('2.jpg')
scan(img7 , (9,9) , 0 , 50 , '1_color_croped_card.jpg')

image = cv2.imread("1_color_croped_card.jpg", 0)
_, thresholded_image_next = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("2_binary_croped_card.jpg", thresholded_image_next)

sobelx = cv2.Sobel(thresholded_image_next, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(thresholded_image_next, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobelx**2 + sobely**2)
angle = np.arctan2(sobely, sobelx)
cv2.imwrite("3_magnitude_croped_card.jpg", magnitude)

threshold_value = 100
edges = np.zeros_like(magnitude)
edges[magnitude > threshold_value] = 255
cv2.imwrite("4_magnitude_tresholded_croped_card.jpg", edges)

# Define the structuring element
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
# Apply the open operator
result_after_fo = cv2.morphologyEx(magnitude, cv2.MORPH_OPEN, structuring_element)
cv2.imwrite("5_magnitude_open_croped_card.jpg", result_after_fo)

# Define the structuring element
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (71, 1))
# Apply the close operator
result_after_c = cv2.morphologyEx(result_after_fo, cv2.MORPH_CLOSE, structuring_element)
cv2.imwrite("6_magnitude_close_croped_card.jpg", result_after_c)

# Define the structuring element
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
# Apply the open operator
result_after_sc = cv2.morphologyEx(result_after_c, cv2.MORPH_OPEN, structuring_element)
cv2.imwrite("7_magnitude_open2_croped_card.jpg", result_after_sc)

# Find connected components and their statistics
list_of_indexes = []
# _, binary_image = cv2.threshold(result_after_sc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if result_after_sc.dtype != np.uint8:
    result_after_sc = result_after_sc.astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result_after_sc)
# Define the minimum area threshold
area_threshold = 1000
# Iterate through connected components
for label in range(1, num_labels):  # Exclude background component (label 0)
    area = stats[label, cv2.CC_STAT_AREA]
    if area > area_threshold:
        # Extract bounding box coordinates
        x, y, width, height = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
                              stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        list_of_indexes.append((x,y,width , height))
        # Draw bounding rectangle on original image (for visualization)
        # if width > 200 and height > 15 and width > height * 9 :
        # cv2.rectangle(image, (x, y), (x + width, y + height), (0, 127, 0), 2)
        # print("draw")
cv2.imwrite("8_magnitude_connected_components_gray_croped_card.jpg", image)

print(list_of_indexes)

def crop_and_replace_boxes(image, boxes):
    # Iterate over the list of boxes
    for box in boxes:
        x, y, width, height = box
        # Crop the box region from the original image
        cropped_box = image[y:y+height, x:x+width]
        # Apply Otsu's thresholding to the cropped box
        _, thresholded_box = cv2.threshold(cropped_box, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thresholded_box = cv2.adaptiveThreshold(cropped_box, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
        # Replace the pixels in the original image with the thresholded box pixels
        image[y:y+height, x:x+width] = thresholded_box
    return image

img_copy = thresholded_image_next.copy()
img_copy = cv2.resize(img_copy, (1500, 1000))
cv2.imwrite("9_magnitude_connected_components_gray_resized_croped_card.jpg", img_copy)

modified_image = crop_and_replace_boxes(img_copy, list_of_indexes)
cv2.imwrite("10_magnitude_connected_components_gray_resized_croped_card.jpg", modified_image)

# Define the structuring element
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Apply the open operator
# modified_image = cv2.morphologyEx(modified_image, cv2.MORPH_DILATE, structuring_element)
cv2.imwrite("11_modified_image_croped_card.jpg", modified_image)

print(list_of_indexes)

def make_templates(template):
    list_candid_temps = []
    # Define the desired number of resize steps
    resize_steps = 10
    # Calculate the resize ratios for both width and height
    width_ratios = np.linspace(1, 0.4, resize_steps)
    height_ratios = np.linspace(1, 0.4, resize_steps)
    # Iterate over the resize ratios
    for width_ratio in width_ratios:
        for height_ratio in height_ratios:
            if abs(width_ratio - height_ratio) == 0:
                # Resize the template
                resized_template = cv2.resize(template, (int(template.shape[1] * width_ratio), int(template.shape[0] * height_ratio)))
                # erosion
                kernel = np.ones((5,5),np.uint8)
                erosion = cv2.erode(resized_template,kernel,iterations = 1)
                # Rotate the resized template
                for angle in range(-5, 5):
                    rotated_template = cv2.warpAffine(resized_template, cv2.getRotationMatrix2D((resized_template.shape[1] / 2, resized_template.shape[0] / 2), angle, 1), (resized_template.shape[1], resized_template.shape[0]))
                    # Add the rotated template to the list of candidates
                    list_candid_temps.append(rotated_template)
               # Rotate the resized template
                for angle in range(-5, 5):
                    rotated_template = cv2.warpAffine(erosion, cv2.getRotationMatrix2D((erosion.shape[1] / 2, erosion.shape[0] / 2), angle, 1), (erosion.shape[1], erosion.shape[0]))
                    # Add the rotated template to the list of candidates
                    list_candid_temps.append(rotated_template)
    return list_candid_temps

def extract_segments(image):
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    matched_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rectangles = []
    for contour in contours:
        if contour is not largest_contour:
            x, y, w, h = cv2.boundingRect(contour)
            rectangles.append((x, y, w, h))

    # Remove nested rectangles
    filtered_rectangles = []
    for i in range(len(rectangles)):
        include_rect = True
        for j in range(len(rectangles)):
            if i != j:
                x1, y1, w1, h1 = rectangles[i]
                x2, y2, w2, h2 = rectangles[j]
                if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                    include_rect = False
                    break
        if include_rect:
            filtered_rectangles.append(rectangles[i])

    # remove rectangles that are too small
    filtered_rectangles = [rect for rect in filtered_rectangles if rect[2] > 20 and rect[3] > 20]

    # remove rectangles that are too large
    filtered_rectangles = [rect for rect in filtered_rectangles if rect[2] < 100 and rect[3] < 100]

    # remove rectangles that width is greater than height
    filtered_rectangles = [rect for rect in filtered_rectangles if rect[2] < rect[3]]

    # cluster rectangles that are close to each other into groups based on their y
    # first, sort rectangles by x and y
    filtered_rectangles.sort(key=lambda x: x[0])
    filtered_rectangles.sort(key=lambda x: x[1])

    clustered_rectangles = []
    for rect in filtered_rectangles:
        if len(clustered_rectangles) == 0:
            clustered_rectangles.append([rect])
        else:
            last_cluster = clustered_rectangles[-1]
            last_rect = last_cluster[-1]
            if abs(rect[1] - last_rect[1]) < 30:
                last_cluster.append(rect)
            else:
                clustered_rectangles.append([rect])


    # sort each cluster by x
    for cluster in clustered_rectangles:
        cluster.sort(key=lambda x: x[0])

    # save top 16 rectangles that have the largest area in each cluster
    filtered_rectangles = []
    for cluster in clustered_rectangles:
        cluster = sorted(cluster, key=lambda x: x[2] * x[3], reverse=True)
        # if len(cluster) > 16:
        #     cluster = cluster[:16]
        filtered_rectangles += cluster


    # Draw the filtered rectangles
    for cluster in clustered_rectangles:
        for (x, y, w, h) in cluster:
            cv2.rectangle(matched_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the image
    plt.figure(figsize=(20, 20))
    # plt.imshow(matched_image)
    # plt.show()
    
    return clustered_rectangles, matched_image
    

def fa_extract_segments(image):

    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    matched_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rectangles = []
    for contour in contours:
        if contour is not largest_contour:
            x, y, w, h = cv2.boundingRect(contour)
            rectangles.append((x, y, w, h))
    # Remove nested rectangles
    filtered_rectangles = []
    for i in range(len(rectangles)):
        include_rect = True
        for j in range(len(rectangles)):
            if i != j:
                x1, y1, w1, h1 = rectangles[i]
                x2, y2, w2, h2 = rectangles[j]
                if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                    include_rect = False
                    break
        if include_rect:
            filtered_rectangles.append(rectangles[i])
    # remove rectangles that are too small
    filtered_rectangles = [rect for rect in filtered_rectangles if rect[2] > 5 and rect[3] > 5]
    # remove rectangles that are too large
    filtered_rectangles = [rect for rect in filtered_rectangles if rect[2] < 100 and rect[3] < 100]
    # remove rectangles that width is greater than height
    # filtered_rectangles = [rect for rect in filtered_rectangles if rect[2] < rect[3]]
    # cluster rectangles that are close to each other into groups based on their y
    # first, sort rectangles by x and y
    filtered_rectangles.sort(key=lambda x: x[0])
    filtered_rectangles.sort(key=lambda x: x[1])
    clustered_rectangles = []
    for rect in filtered_rectangles:
        if len(clustered_rectangles) == 0:
            clustered_rectangles.append([rect])
        else:
            last_cluster = clustered_rectangles[-1]
            last_rect = last_cluster[-1]
            if abs(rect[1] - last_rect[1]) < 30:
                last_cluster.append(rect)
            else:
                clustered_rectangles.append([rect])
    # sort each cluster by x
    for cluster in clustered_rectangles:
        cluster.sort(key=lambda x: x[0])
    # save top 16 rectangles that have the largest area in each cluster
    filtered_rectangles = []
    for cluster in clustered_rectangles:
        cluster = sorted(cluster, key=lambda x: x[2] * x[3], reverse=True)
        # if len(cluster) > 16:
        #     cluster = cluster[:16]
        filtered_rectangles += cluster
    # Draw the filtered rectangles
    for cluster in clustered_rectangles:
        for (x, y, w, h) in cluster:
            cv2.rectangle(matched_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # Display the image
    plt.figure(figsize=(20, 20))
    # plt.imshow(matched_image)
    # plt.show()
    return clustered_rectangles, matched_image
    
def template_matching(filtered_rectangles, templates):
    # crop rectangles and compare with candidates
    scores = {(x, y, w, h): {template: 0 for template in templates} for (x, y, w, h) in filtered_rectangles}
    for (x, y, w, h) in filtered_rectangles:
        crop_img = image[y:y+h, x:x+w]
        for template in templates:
            max_score = 0
            for candid_template in templates[template]:
                # make size equal
                candid_template = cv2.resize(candid_template, (crop_img.shape[1], crop_img.shape[0]))
                # compare in CCOEFF_NORMED
                res = cv2.matchTemplate(crop_img, candid_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val > max_score:
                    max_score = max_val
            scores[(x, y, w, h)][template] = max_score

    # find max score
    max_scores = {}
    for (x, y, w, h) in filtered_rectangles:
        # set key of max_scores
        number = -1
        max_score = 0
        for template in templates:
            if scores[(x, y, w, h)][template] > max_score:
                max_score = scores[(x, y, w, h)][template]
                number = template
        max_scores[(x, y, w, h)] = number if max_score > 0.1 else None

    # sort by x
    filtered_rectangles = sorted(filtered_rectangles, key=lambda x: x[0])

    data = ""
    # draw number
    for (x, y, w, h) in filtered_rectangles:
        char = max_scores.get((x, y, w, h))
        if char is None:
            continue
        cv2.putText(matched_image, max_scores[(x, y, w, h)], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if char == 'slash.png':
            char = '/.png'
        data += char.split('.')[0]


    # Display the image
    # plt.figure(figsize=(20, 20))
    # plt.imshow(matched_image)
    # plt.show()

    return data


def template_matching(filtered_rectangles, templates):
    # crop rectangles and compare with candidates
    scores = {(x, y, w, h): {template: 0 for template in templates} for (x, y, w, h) in filtered_rectangles}
    for (x, y, w, h) in filtered_rectangles:
        crop_img = image[y:y+h, x:x+w]
        for template in templates:
            max_score = 0
            for candid_template in templates[template]:
                # make size equal
                candid_template = cv2.resize(candid_template, (crop_img.shape[1], crop_img.shape[0]))
                # compare in CCOEFF_NORMED
                res = cv2.matchTemplate(crop_img, candid_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val > max_score:
                    max_score = max_val
            scores[(x, y, w, h)][template] = max_score

    # find max score
    max_scores = {}
    for (x, y, w, h) in filtered_rectangles:
        # set key of max_scores
        number = -1
        max_score = 0
        for template in templates:
            if scores[(x, y, w, h)][template] > max_score:
                max_score = scores[(x, y, w, h)][template]
                number = template
        max_scores[(x, y, w, h)] = number if max_score > 0.1 else None

    # sort by x
    filtered_rectangles = sorted(filtered_rectangles, key=lambda x: x[0])

    data = ""
    # draw number
    for (x, y, w, h) in filtered_rectangles:
        char = max_scores.get((x, y, w, h))
        if char is None:
            continue
        cv2.putText(matched_image, max_scores[(x, y, w, h)], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if char == 'slash.png':
            char = '/.png'
        data += char.split('.')[0]


    # Display the image
    # plt.figure(figsize=(20, 20))
    # plt.imshow(matched_image)
    # plt.show()

    return data

def generate_templates(dataset_name):
    templates_img = ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png', 'slash.png']
    # make_candidates
    templates = {template: [] for template in templates_img}
    for template in templates_img:
        i = 1
        while True:
            try:
                imread_template = cv2.imread('./'+ dataset_name + '_' + str(i) + "/" + template, 0)
                templates[template] += make_templates(imread_template)
            except:
                break
            i += 1

    return templates


# check if card is bank-card or national-card
def is_national_card(image):
    allah = image[700:1000 , 0:300]
    allah_template = cv2.imread('sun.png', 0)
    # template matching
    res = cv2.matchTemplate(allah, allah_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.3
    # show hit map
    loc = np.where(res >= threshold)
    # print max score
    # print(np.max(res))
    if len(loc[0]) > 0:
        return True
    else:
        return False
        
        
image = modified_image
    
if is_national_card(modified_image):
    templates = generate_templates('Fa_dataset')
    image = image[200:650, 800:1250]
    clustered_rectangles, matched_image = fa_extract_segments(image)
    
    for cluster in clustered_rectangles:
        phrase = template_matching(cluster, templates)
        if "/" in phrase:
            print("expir_date: ", phrase)
            continue
            slash_index = phrase.index("/")
            expir_date = phrase[slash_index - 2: slash_index + 3]
            if len(phrase)>=7 and slash_index >= 4:
                print("expir_date: ", expir_date)
        elif len(phrase) == 10:
            national_id = phrase
            print("national_id: ", national_id)
    
else:
    templates = generate_templates('dataset')
    clustered_rectangles, matched_image = extract_segments(image)

    for cluster in clustered_rectangles:
        phrase = template_matching(cluster, templates)
        if "/" in phrase:
            slash_index = phrase.index("/")
            expir_date = phrase[slash_index - 2: slash_index + 3]
            if len(phrase)>=7 and slash_index >= 4:
                print("expir_date: ", expir_date)
        elif len(phrase) == 16:
            card_number = phrase
            print("card_number: ", card_number)


cv2.imwrite('matched_image.png', matched_image)











