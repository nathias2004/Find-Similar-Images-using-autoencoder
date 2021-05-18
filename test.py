# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:24:20 2020

@author: Venkata Sainath
"""


import os
import numpy as np
from src.CV_IO_utils import read_imgs_dir
from src.CV_transform_utils import apply_transformer
from src.CV_transform_utils import resize_img, normalize_img
from src.autoencoder import AutoEncoder
from PIL import Image
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
from sklearn.metrics.pairwise import cosine_similarity
import shutil
from sklearn.neighbors import NearestNeighbors
#Load the embeddings and cluster them

X = np.load("Embeddings.npy")
print(X.shape)
  
#Building and fitting the model Clustering
kmeanModel = KMeans(n_clusters=7).fit(X) 
kmeanModel.fit(X)     
Labels = kmeanModel.labels_  


    
test_dir = os.path.join(os.getcwd(), "data", "test")
if not os.path.exists(test_dir):
    raise Exception("Create a data/test directory with images")
    
extensions = [".jpg"]
args = [os.path.join(test_dir, filename)
            for filename in os.listdir(test_dir)
            if any(filename.lower().endswith(ext) for ext in extensions)]

#Resize the images to low pixels for computation purposes
if len(args) == 0:
    raise Exception("Put test images in the test directory")

cleaned_dir = os.path.join(os.getcwd(), "data", "test", "cleaned")
if not os.path.exists(cleaned_dir):
    os.makedirs(cleaned_dir)
#Cleaned images dir is created



generation_dir = os.path.join(os.getcwd(), "data", "generated")
if not os.path.exists(generation_dir):
    os.makedirs(generation_dir)



#Load the model
modelName = "convAE"
shape_img = (64,64,3)
outDir = os.path.join(os.getcwd(), "output")
info = {
    "shape_img": shape_img,
    "autoencoderFile": os.path.join(outDir, "{}_autoecoder.h5".format(modelName)),
    "encoderFile": os.path.join(outDir, "{}_encoder.h5".format(modelName)),
    "decoderFile": os.path.join(outDir, "{}_decoder.h5".format(modelName)),
}
model = AutoEncoder(modelName, info)
model.set_arch()
model.load_models(loss="binary_crossentropy", optimizer="adam")

shape_img_resize = shape_img
input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])


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



N = int(input("Number of images to be generated: "))

idx = 0
    
for filename in args:
    idx = idx + 1
    fp = filename
    im = Image.open(fp,mode="r")
    im1 = im.resize((64,64))
    fp1 = "data/test/cleaned/temp"+".jpg"
    im1.save(fp1)
    
    
    image_dir = os.path.join(os.getcwd(), "data", "generated", str(idx))    
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    fp2 = image_dir+"/query.jpg"
    im.save(fp2)
    imgs_test = read_imgs_dir(cleaned_dir, extensions, parallel=False)
    imgs_test_transformed = apply_transformer(imgs_test, transformer, parallel=False)

    # Convert images to numpy array
    X_test = np.array(imgs_test_transformed).reshape((-1,) + input_shape_model)
    E_test = model.predict(X_test)
    E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))
    #print(kmeanModel.predict(E_test_flatten))
    
    predicted_label = kmeanModel.predict(E_test_flatten)
    Neighbours = []
    for i in range(0,len(Labels)):
        if(Labels[i] == predicted_label):
            #print(E_test_flatten[0].shape,X[i].shape)
            distance = cosine_similarity(E_test_flatten,[X[i]])
            Neighbours.append([distance[0][0],i])
    Neighbours = sorted(Neighbours,key = lambda x: x[0], reverse=True) 
    
    for neighbour in Neighbours[0:N]:
        index = neighbour[1]
        fp = "dataset/"+str(index)+".jpg"
        im = Image.open(fp,mode="r")
        fp1 = image_dir+"/"+str(index)+".jpg"
        im.save(fp1)
    
    
        
        
    
            
    
    

    
    
    
