from keras.callbacks import ModelCheckpoint, EarlyStopping
from models import lstm_model
from data import DataSet
import os.path
import sys
import keras.backend.tensorflow_backend as tfback
import tensorflow as tf

''' Initialise gpu session '''
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

def _get_available_gpus():
    if tfback._LOCAL_DEVICES == None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
tfback._get_available_gpus = _get_available_gpus
tfback._get_available_gpus
tf.config.list_logical_devices()


''' Training function '''
def train(sequence_length,
          image_shape,
          batch_size,
          nb_epoch):
    
    filepath = os.path.join('data', 'checkpoints', 'ConvLSTM.{epoch:03d}-{mse:.5f}.hdf5')
    
    # helper: save model 
    checkpointer = ModelCheckpoint(filepath = filepath,
                                    monitor='mse',
                                    verbose=2,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto')
    
    # helper: stop training when model converges
    early_stopper = EarlyStopping(monitor='mse',
                              min_delta=0,
                              patience=10,
                              restore_best_weights=True)
    
    # Get the training data
    data = DataSet(
        sequence_length=sequence_length,
        image_shape=image_shape)
    
    # Get samples per epoch.
    # Multiply by 0.7 to estimate how much data is the train set
    steps_per_epoch = (len(data.data) * 0.70) // batch_size
    # Multiply by 0.3 to estimate how much data is the validation set
    validation_steps = (len(data.data) * 0.30) // batch_size

    # Data generators
    generator = data.frame_generator(batch_size, 'train', augment = True)
    val_generator = data.frame_generator(batch_size, 'test', augment = False)

    # Get the model
    model = lstm_model()

    # Train the model
    history = model.fit_generator(generator=generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=nb_epoch,
                            verbose=0,
                            callbacks=[early_stopper, checkpointer],
                            validation_data=val_generator,
                            validation_steps=validation_steps)

    # Close GPU session
    session.close()


def main() -> object:
    
    """Training settings. Set before training"""
    if (len(sys.argv) == 5):
        sequence_length = int(sys.argv[1]) # default 30
        image_height = int(sys.argv[2]) # default 112
        image_width = int(sys.argv[3]) # default 112
        batch_size = int(sys.argv[4]) # default 2
        nb_epoch = int(sys.argv[5]) # default 1000

    else:
        print ("Usage: python train.py sequence_length image_height image_width batch_size number_of_epochs")
        print ("Example: python train.py 30 112 112 2 1000")
        exit (1)
        
    image_shape = (image_height, image_width, 3)

    checkpoints_dir = os.path.join('data', 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    train(sequence_length,image_shape,batch_size,nb_epoch)


if __name__ == '__main__':
    main()
