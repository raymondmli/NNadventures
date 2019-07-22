#!/usr/bin/env python
import cv2

class SimplePreprocessor:
  def __init__(self,width,height,inter=cv2.INTER_AREA):
    self.width = width
    self.height = height
    self.inter = inter
    
  #resizes image to fixed size 
  def preprocess(self,image):
    return cv2.resize(image,(self.width, self.height),
                      interpolation=self.inter) 
    