# utils.py
import cv2
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.transformers import ImageResizer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, model_path: str, vocab: list, *args, **kwargs):
        super().__init__(model_path=model_path, *args, **kwargs)
        self.vocab = vocab
        self.input_name='input'
    def predict(self, image: np.ndarray,language="English"):
        
        if language == "Devanagari":
            image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])
            image_pred = np.expand_dims(image, axis=0).astype(np.float32)
            preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
            text = ctc_decoder(preds, self.vocab)[0]
            return text
        
        image = cv2.resize(image, (1408, 96))
        # Ensure the image has 3 color channels
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # # Add an extra dimension for the batch size
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.vocab)[0]
        return text
    
def chain_line(img):
    """
    Finds the contours (boundaries) of the objects in a given image.

    Args:
        img (numpy.ndarray): The input image as a numpy array.

    Returns:
        list: A list of contours, where each contour is represented as a list of (x, y) coordinates.

    Notes:
        - This function performs the following steps:
            1. Applies binary thresholding to the input image to create a black-and-white image.
            2. Dilates the binary image to expand the white regions.
            3. Finds the contours (boundaries) of the white regions in the dilated image using the Chain Approximation Simple algorithm.
        - The function returns a list of the detected contours, which can be useful for tasks like object detection, shape analysis, or other image processing applications.
    """
    _, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresholded, np.ones((5, 200), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def chain_word(img):
    _, thresholded = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresholded, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def otsu_line(img):
    """
    Detects and returns the contours of text lines in an image using the Otsu's method for thresholding.

    Args:
        img (numpy.ndarray): The input image as a numpy array.

    Returns:
        list: A list of contours, where each contour is represented as a list of (x, y) coordinates.

    Notes:
        - This function performs the following steps:
            1. Applies Gaussian blurring to the input image to reduce noise.
            2. Applies Otsu's method for automatic thresholding, creating a binary image.
            3. Applies a closing morphological operation using a rectangular kernel to connect the text lines.
            4. Finds the contours of the connected text lines using the OpenCV `findContours` function.
        - The function returns a list of the detected contours, which can be useful for tasks like text extraction, layout analysis, or other document processing applications.
    """
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # selected a kernel with more width so that we want to connect lines
    kernel_size = (15, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Perform the closing operation: Dilate and then close
    bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    
    # Find contours for each text line
    contours, _ = cv2.findContours(bw_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def detect_contours(image, edge_detection_method="chain_line", predict=False, model=None, language="English"):
    def process_contours(image, contours, color=(0, 255, 0)):
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            roi_row = roi.shape[0]
            roi_col = roi.shape[1]
             # Show ROI
            if(roi_row>1000 or roi_row<=30):
                continue
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    def resize_image(image, max_width=500, max_height=500):
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    color_image = image.copy()
        # Check the number of channels in the image
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            # Convert image from BGR to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            # Convert image from BGRA to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        # The image is already grayscale
        gray = image
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        

    methods = {
    "chain_line": chain_line,
    "chain_word": chain_word,
    "otsu_line": otsu_line
    }

    contours = methods.get(edge_detection_method)(gray)
    process_contours(color_image, contours)
    
    if predict:
        predicted_texts = []
        for contour in contours:
            
            x, y, w, h = cv2.boundingRect(contour)
            
            roi = color_image[y:y+h, x:x+w]
            roi_row = roi.shape[0]
            roi_col = roi.shape[1]
             # Show ROI
            if(roi_row>1000 or roi_row<=50):
                continue
            
            prediction = model.predict(roi,language=language)  # make prediction here
            predicted_texts.append((x, y, prediction))
            
        # Sort the list based on y-coordinate, then x-coordinate
        sorted_predictions = sorted(predicted_texts, key=lambda x: (x[1], x[0]))
        
        final_predictions = [x[2] for x in sorted_predictions]
        final_predictions = [x[2].lstrip('I') for x in sorted_predictions]
        print(final_predictions)
        
        return final_predictions
        

    return resize_image(color_image)