# Find-Similar-Images-using-autoencoder
You are provided with a dataset of ~5k 512x512 images, your program should accept an 512x512 input image and return N images from the provided dataset similar to the input image.

## Training:
->Put the original Images in "dataset" folder
->remove all the images in "data" folder
->run python "image_retrieval.py"


## Test:
->put the test Images in "data/test"
->run "test.py"
->Give the number of sample required(N)
->The generated images will be saved in "data/generated" folder seperately


## Clustering:
->run "Clustering.py"  to see the elbow curve


## Brief Implementation:

->The sizes of the images are large and processing large images is computationally expensive
            ->To tackle this two approaches can be tried
                1)Cropping each image into multiple images and training
                2)Resizing/Reducing the pixel sizes
->I followed by resizing the image
->A Convolution Auto Encoder is trained and Embeddings are saved in Embeddings.npy
->Now to find out number of optimal clusters, Elbow curve is drawn by experimenting with different K's
using distortion and inertia
->kmeans clustering is used
->Using the elbow curves number of clusters is found as 7
->During inferencing a test image...i found the cluster assignment and searched for the nearest images to 
the test image in the respective cluster


Requirements:
Tensorflow
sklearn
PIL
numpy
matplotlib


References:
The basic code is inspired from https://github.com/ankonzoid/artificio



