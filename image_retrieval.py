"""
Created on Fri Dec 11 21:22:21 2020

@author: Venkata Sainath
"""
import os
import numpy as np
from src.CV_IO_utils import read_imgs_dir
from src.CV_transform_utils import apply_transformer
from src.CV_transform_utils import resize_img, normalize_img
from src.autoencoder import AutoEncoder
from PIL import Image


train_dir = os.path.join(os.getcwd(), "data", "train")
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

#Resize the images to low pixels for computation purposes
for i in range(0,4738):
    fp = "dataset/"+str(i)+".jpg"
    im = Image.open(fp,mode="r")
    im1 = im.resize((64,64))
    fp1 = "data/train/"+str(i)+".jpg"
    im1.save(fp1)
print("Resizing completed")

modelName = "convAE"
trainModel = True
parallel = False  # use multicore processing

# Make paths
dataTrainDir = os.path.join(os.getcwd(), "data", "train")
outDir = os.path.join(os.getcwd(), "output")
if not os.path.exists(outDir):
    os.makedirs(outDir)

# Read images
extensions = [".jpg"]
print("Reading train images from '{}'...".format(dataTrainDir))
imgs_train = read_imgs_dir(dataTrainDir, extensions, parallel=parallel)
shape_img = imgs_train[0].shape
print("Image shape = {}".format(shape_img))

# Build models
# Set up autoencoder
info = {
    "shape_img": shape_img,
    "autoencoderFile": os.path.join(outDir, "{}_autoecoder.h5".format(modelName)),
    "encoderFile": os.path.join(outDir, "{}_encoder.h5".format(modelName)),
    "decoderFile": os.path.join(outDir, "{}_decoder.h5".format(modelName)),
}
model = AutoEncoder(modelName, info)
model.set_arch()

shape_img_resize = shape_img
input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])
n_epochs = 20


# Print some model info
print("input_shape_model = {}".format(input_shape_model))
print("output_shape_model = {}".format(output_shape_model))

# Apply transformations to all images
class ImageTransformer(object):

    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

transformer = ImageTransformer(shape_img_resize)
print("Applying image transformer to training images...")
imgs_train_transformed = apply_transformer(imgs_train, transformer, parallel=parallel)

# Convert images to numpy array
X_train = np.array(imgs_train_transformed).reshape((-1,) + input_shape_model)
print(" -> X_train.shape = {}".format(X_train.shape))

# Train (if necessary)
if trainModel:
    model.compile(loss="binary_crossentropy", optimizer="adam")
    model.fit(X_train, n_epochs=n_epochs, batch_size=32)
    model.save_models()
else:
    model.load_models(loss="binary_crossentropy", optimizer="adam")

# Create embeddings using model
print("Inferencing embeddings using pre-trained model...")
E_train = model.predict(X_train)
E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
print(" -> E_train.shape = {}".format(E_train.shape))
print(" -> E_train_flatten.shape = {}".format(E_train_flatten.shape))
np.save("Embeddings",E_train_flatten)