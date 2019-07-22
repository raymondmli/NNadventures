#!/usr/bin/env python
# import the necessary packages
import sys 
sys.path.append("/Users/raymondli/comp/deepLearning/6441project/pyimagesearch")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import simplepreprocessor as sp 
from datasets import simpledatasetloader as sdl
from preprocessing import imagetoarraypreprocessor as iap
from imutils import paths
import argparse

sp = sp.SimplePreprocessor(32, 32)
iap = iap.ImageToArrayPreprocessor()
sdl = sdl.SimpleDatasetLoader(preprocessors=[sp,iap])