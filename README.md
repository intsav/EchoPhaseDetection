# EchoPhaseDetection
Code files to accompany the paper *"Multibeat Echocardiographic Phase Detection Using Deep Neural Networks".*

### Step 1
Place the videos from your dataset in the following directories:

> | /data/test
> >		...
> | /data/train
> >		...

### Step 2
Generate the target labels and place in the data folder with the filename 'labels.csv'.

Csv format:

| Video frame name | Label |
| ------------- | ------------- |
| video1_1_1  | 0.92  |
| video1_1_2 | 0.83  |

### Step 3

Produce a csv file containing the filenames of your training and validation sets, save to the data folder with the name 'video_info.csv'.

Csv format:

| Dataset  | Video filename | Number of frames |
| ------------- | ------------- | ------------- |
| test  | video1_1  | 30 |
| train | video1_2  | 30 |

### Step 4
Run train.py with the following args:   sequence_length image_height   image_width   batch_size   number_of_epochs

For example:

`	$ python train.py 30 112 112 2 1000`

### Step 5
Generate predictions from `data/predictions/run_predictions.py` using the saved model with the best weights from training.
