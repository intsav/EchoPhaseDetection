import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from scipy.signal import find_peaks

''' Data management class to predict videos of arbitrary length '''
class Predict():
    
    def __init__(self, fpath, seqlen, stride):
        self.file_path = fpath
        self.sequence_length = seqlen
        self.stride = stride
        self.frames = self.get_frames()
        self.num_padded_frames = 0
    
    # Get frames from input video sequence
    def get_frames(self):
        capture = cv2.VideoCapture(self.file_path)
        frames=[]
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        num_frames = len(frames)
        padded_frame = np.zeros([112,112,3], dtype='uint8')
        # If not divisible by sequence length, add padded frames
        num_pads = num_frames%self.sequence_length
        if num_pads>0:
          new_frames = self.sequence_length-num_pads
        else:
          new_frames=0
        self.num_padded_frames = new_frames
        if num_pads != 0:
          for i in range(new_frames):
              frames.append(padded_frame)   
        return frames

    # Normalise input data, return as a sequence
    def get_image_sequence(self, frames):
        sequence = []
        for img in frames:
          img_arr = img_to_array(img)
          x = (img_arr / 255.).astype(np.float32)
          sequence.append(x)
        return sequence

    # Chunk data using sliding window
    def get_chunked_sequence(self, sequence):
        chunked_sequence = []
        for i in range(0, len(sequence), self.stride):
            new = sequence[i:i + self.sequence_length]
            if len(new)==self.sequence_length:
                chunked_sequence.append(new)
        return chunked_sequence
    
    def get_predictions(predictions):
        # invert predictions to find_peaks ES frames
        inverted = predictions * -1
    
        ED = find_peaks(predictions,prominence=0.05,distance=18)
        ES = find_peaks(inverted,prominence=0.03,distance=15)
    
        # extract predictions and convert to list
        edLst = np.array(ED).tolist()
        esLst = np.array(ES).tolist()
        ED=edLst[0]
        ES=esLst[0]
        ED = ED.tolist()
        ES = ES.tolist()
    
        # get index of max prediction from first 5
        beg = predictions[:5]
        beg = beg.tolist()
        a = beg.index(max(beg))
    
        # Try to work out if it is a peak and if already in ED
        if a == 0:
          if not a in ED:
            ED.append(a)
            ED.sort()
        else:
          try:
            p = beg[a-1]
            n = beg[a+1]
            if a>p and a>n:
              if not a in ED:
                ED.append(a)
                ED.sort()
          except:
            pass
        
        return ED, ES
