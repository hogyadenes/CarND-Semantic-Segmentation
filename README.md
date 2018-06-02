# Semantic Segmentation
### Introduction
In this project, the task was to label the pixels of a road in images using a Fully Convolutional Network (FCN).

The project mainly consisted of implementing the FCN-8 and tuning its hyperparameters. The pretrained VGG network was loaded from a data file and trained using the labeled Kitti dataset.

For training, I decided to use an l2 kernel regularizer and an Adam optimizer (it has fewer hyperparameters).

The training and testing was done on my home computer (i7 processor and Geforce GTX980ti).

### Hyperparameters

I had to tune the following hyperparameters:
1. Learning rate
2. Epochs
3. Batch size
4. Keep probability
5. Regularization

After a long experimenting phase, the final parameter set was ended up being the following:

1. LEARNING_RATE = 5e-4
2. EPOCHS = 30
3. BATCH_SIZE = 5
4. KEEP_PROB = 0.75
5. REGULARIZER = 1e-3

The final image set is [runs/1527873951.8560643](https://github.com/hogyadenes/CarND-Semantic-Segmentation/tree/master/runs/1527873951.8560643). The final log file (containing the average loss per epoch) is [here](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/_)

Some example images from the final result:
![1](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/um_000002.png)
![2](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/um_000004.png)
![3](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/um_000010.png)
![4](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/um_000014.png)
![5](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/um_000023.png)
![6](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/um_000029.png)
![7](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/um_000068.png)
![8](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/um_000074.png)
![9](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/um_000075.png)
![10](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/umm_000017.png)
![11](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/umm_000020.png)
![12](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/umm_000092.png)
![13](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/uu_000040.png)
![14](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/uu_000057.png)
![15](https://github.com/hogyadenes/CarND-Semantic-Segmentation/blob/master/runs/1527873951.8560643/uu_000028.png)

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


