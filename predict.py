import numpy as np 
import pytesseract
import cv2
import matplotlib.pyplot as plt
import os
import re
import math

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def levenshtein(s, t):
    """ 
        Calculates levenshtein distance between two strings.
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions

    return distance[len(s)][len(t)]

def get_location_from_image(img):
    """
    img: image of tracetogether screenshot
    default_tesseract_cmd_path: Path of executable for tesseract
    """

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna()

    # data['top'] = data['top'].map(lambda x: int(math.ceil(x / 100.0) * 100))
    data['text'] = data.groupby(['block_num'])['text'].transform(lambda x: ' '.join(x))
    data = data.drop_duplicates(subset=['text'])
    data = data.sort_values(by=['top', 'left'])

    text_lines = data['text'].tolist()

    # Preprocessing string
    for i in range(len(text_lines)):
        text_lines[i] = re.sub('[!@#$_|&?]', '', text_lines[i]).strip()

    text_lines = [i.strip() for i in text_lines if len(i) > 5]

    index_of_location = -2
    for i in range(len(text_lines)):
        # 24 Jan, 2:21PM or 31 Feb
        if re.match(r'\d{1,2}\s*\w{3}[\,|\.]\s*\d{1,2}[\.|\:|\,]\d{2}\s*(AM|PM)', text_lines[i]):
            index_of_location = i + 1
            break
    
    if index_of_location < 0:
        for i in range(len(text_lines)): 
            if 'GOVTECH' in text_lines[i].upper():
                index_of_location = i - 1
                break
        
    if index_of_location < 0 or index_of_location >= len(text_lines):
        return "NOT FOUND"
        
    return text_lines[index_of_location].strip()

def get_bounding_box_of_text(img, text):
    """
    Gets the bounding box coordinates of the location on the phone

    img: RGB Image
    text: text to find bounding box for

    Returns (x_coordinate, y_coordinate, width_of_bounding_box, height_of_bounding_box)
    """
    img_height, img_width = img.shape[:2]

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna()
    data['text'] = data.groupby('block_num')['text'].transform(lambda x: ' '.join(x))
        
    # data = data[data['text'].str.contains(text, regex=False)]
    data = data[data.apply(lambda row: len(row['text'].strip()) > 5 and 
        ((row['text'] in text or text in row['text']) or 
        (levenshtein(row['text'], text) / len(text) < 0.2))
        , axis=1)]

    x1 = img_width
    y1 = img_height
    x2 = 0
    y2 = 0

    for index, row in data.iterrows():
        x_current_1, y_current_1 = row['left'], row['top']
        x_current_2, y_current_2 = row['left'] + row['width'], row['top'] + row['height'] 

        if x_current_1 < x1:
            x1 = x_current_1
        
        if y_current_1 < y1:
            y1 = y_current_1

        if x_current_2 > x2:
            x2 = x_current_2

        if y_current_2 > y2:
            y2 = y_current_2

    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1

    return (x, y, w, h)

def is_vaccinated_screenshot_color(r, g, b):
    return (r < g and r < b) or ((r < 180) and (r < g and r < b) and (g > 100 or b > 100))

def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
    dilate = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=1)
    thresh = cv2.threshold(dilate, 150, 150,cv2.THRESH_BINARY)[1]
    erode = cv2.erode(thresh, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.addWeighted( erode, 50, erode, 0, 0)

def check_if_vaccinated_and_at_correct_location(img, location):
    """
    Given an image and a location, check if user is vaccinated and is at the correct location

    img: Image to check (in RGB format)
    location: String of location user is supposed to be in
    """
    preprocessed_img = preprocess_img(img)
    img_location = get_location_from_image(preprocessed_img)

    # Location detected by image may be a little bit off. Use fuzzy word matching to allow some leeway
    # If the entire location is within the predicted location (Extra words), we consider a match
    # If not, use fuzzy word matching to ensure that the location is correct
    if location.lower().strip() not in img_location.lower().strip() and levenshtein(img_location.lower().replace(' ', ''), location.lower().replace(' ', '')) / len(location) > 0.05:
        return False

    x, y, w, h = get_bounding_box_of_text(preprocessed_img, img_location)

    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    y += 4 * h
    h *= 4

    roi = img[y:y+h, x:x+w]
    r, g, b, a = cv2.mean(roi)
    r = int(r)
    g = int(g)
    b = int(b)

    return is_vaccinated_screenshot_color(r, g, b)

if __name__ == "__main__":
    img = cv2.imread("screenshots/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(check_if_vaccinated_and_at_correct_location(img, "QB HOUSE JURONG POINT"))
