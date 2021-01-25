# EchoPhaseDetection
Code files to accompany the paper *"Multibeat Echocardiographic Phase Detection Using Deep Neural Networks".*

### Step 1
Place the videos from your dataset in the following directories:

> >	| /data/test
> > >		...
> >	| /data/train
> > >		...

### Step 2
Generate the target labels and place in the data folder with the filename 'labels.csv'.

Produce a csv file containing the filenames of your training and validation sets, save to the data folder with the name 'video_info.csv'.

### Step 3
Run train.py with the following args: sequence_length image_height image_width batch_size number_of_epochs
`	$ python train.py 30 112 112 2 1000`

