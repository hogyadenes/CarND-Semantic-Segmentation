# Semantic Segmentation
### Introduction
In this project, the task was to label the pixels of a road in images using a Fully Convolutional Network (FCN).

The project mainly consisted of implementing the FCN-8 and tuning its hyperparameters. The pretrained VGG network was loaded from a data file and trained using the labeled Kitti dataset.

For training, I decided to use an l2 kernel regularizer and an Adam optimizer (it has fewer hyperparameters).

The training and testing was done on my home computer (i7 processor and Geforce GTX980ti) and one

### Hyperparameters

I had to tune the following hyperparameters:
1. Learning rate
2. Epochs
3. Batch size
4. Keep probability
5. Regularization

The final parameter is is the following:
LEARNING_RATE = 5e-4
EPOCHS = 30
BATCH_SIZE = 5
KEEP_PROB = 0.75
REGULARIZER = 1e-3







### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py
```


