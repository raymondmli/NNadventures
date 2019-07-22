#!/usr/bin/env python
from keras.preprocessing.image import img_to_array

#just  calls the keras function img_to_array
#probably a little unnecessary
class ImageToArrayPreprocessor:
    def __init__(self,dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self,image):
        return img_to_array(image,data_format=self.dataFormat)
