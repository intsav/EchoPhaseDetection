import pandas as pd
import numpy as np
import os.path
import threading
import random
from processor import process_image
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image

''' Data management class '''

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

# Get the data
class DataSet():
    def __init__(self, seq_length, image_shape):
        self.seq_length = seq_length
        self.data = self.get_data()
        self.labels = self.get_label_data()
        self.image_shape = image_shape
  
    @staticmethod
    def get_data(): # Image data
        data = pd.read_csv('data/video_info.csv', header=None)
        data = data.values.to_list()
        return data
    
    @staticmethod
    def get_label_data(): # Target data
        label_data = pd.read_csv('data/labels.csv', header=None)
        label_data = label_data.values.to_list()
        return label_data
    
    # Split data into train and validation sets
    def split_train_test(self):
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test
    
    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, augment=False):
        augment = augment
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test
        print(f"Creating {train_test} generator with {len(data)} samples")
        while 1:
            X, y = [], []
            for _ in range(batch_size):
                sequence = None # Ensure sequence is empty
                sample = random.choice(data) # Choose random sample from data
                frames = self.get_frames_for_sample(sample)
                sequence = self.build_image_sequence(frames)
                if augment == True:
                    sequence = self.augmentor(sequence) # Augmentation
                lab = []
                # get labels
                for frame in frames:
                    frame_name = self.get_filename_from_image(frame)
                    for row in self.labels:
                        if row[0] == frame_name:
                            lab.append(row[1])
                        else:
                            next
                X.append(sequence)
                y.append(lab)
                
            X = np.array(X)
            y = np.array(y)
            
            yield X, y

    ''' Augment training data '''        
    def augmentor(self,images):
        
        seq = iaa.Sequential([
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                   rotate=(-10, 10)),
        iaa.Crop(percent=(0, 0.1))],
            random_order=True)
        
        images_aug = seq(images=images)
        
        return images_aug
                       
    ''' Return sequence of images in each video sample '''
    def build_image_sequence(self, frames):
        return [process_image(x, self.image_shape) for x in frames]
    
    ''' Return sequence of images from video sample '''
    def get_frames_by_filename(self, filename):
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError(f"Couldn't find sample: {filename}")

        frames = self.get_frames_for_sample(sample)
        frames = self.rescale_list(frames, self.seq_length)
        sequence = self.build_image_sequence(frames)

        return sequence
    
    @staticmethod
    def get_frames_for_sample(sample):
        path = f'data/{sample[0]}'
        filename = sample[1]
        ims = f"{path}/{filename}_" + '*.jpg'
        names = []
        for i in ims:
            im=i.split('_')
            im = im[3]
            im=im.split('.')
            im = int(im[0])
            names.append(im)
        x = sorted(names)
        sortedSamples=[]
        for xi in x:
            f = f'{path}/{filename}_{xi}.jpg'
            sortedSamples.append(f)
        return sortedSamples

    @staticmethod
    def get_filename_from_image(frame):
        parts = frame.split('/')
        name=parts[2]
        name=name.replace('.jpg', '')
        return name

