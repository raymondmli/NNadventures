#!/usr/bin/env python
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from pyimagesearch.preprocessing import imagetoarraypreprocessor as iap
from pyimagesearch.datasets import simpledatasetloader as sdl
from pyimagesearch.preprocessing import simplepreprocessor as spp 
from pyimagesearch.nn import shallownet as sn 
from keras.optimizers import SGD
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse

#construct an argument parser to parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

# load dataset, scale pixel intensities between 0 and 1 
sp = spp.SimplePreprocessor(32, 32)
ip = iap.ImageToArrayPreprocessor()
sd = sdl.SimpleDatasetLoader(preprocessors=[sp,ip])
(data,labels) = sd.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0 

# partition data into train and test 
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = sn.ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    batch_size=32, epochs=100, verbose=1)

 # evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=["cat", "dog", "panda"]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()