from keras.preprocessing.image import img_to_array, load_img
import numpy as np

""" Process each image in sequence and return as a numpy array """
def process_image(image, target_shape):
    # Load image
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))
    
    # Convert to numpy array, normalize and return
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)
        
    return x
