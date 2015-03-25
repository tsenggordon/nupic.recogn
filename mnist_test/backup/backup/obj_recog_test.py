#usr/bin/env python

import os
import Image
import numpy as np
from obj_recog import Object_Recognition as obj_sub
import dataset_readers as data_reader
import image_encoders as img_encoder
from classifiers import exactMatch
from classifiers import KNNClassifier
class object_recognition_testbench():
   
   def __init__(self,
                inputShape = (32, 32),
                columnDimensions = (16, 16),
                activeRate = 0.1, precision = 8):
      self.obj_module = obj_sub(inputShape, columnDimensions, \
                                activeRate, precision)
      self.clf = KNNClassifier()
      self.SDRList = []
      self.testResultTag = []   

   def buildTagsTable(self, tags, verbosity = 0):
      self.tagsTable = []
      for tag in tags:
         if tag not in self.tagsTable:
            self.tagsTable.append(tag)
      if verbosity:
         print self.tagsTable
   def changeTrainingImg(self, path, verbosity = 0):
      #img = Image.open(path).convert('LA')
      self.trainImages, self.trainFilenames, self.trainTags = data_reader.getImagesAndTags(path)
      self.buildTagsTable(self.trainTags, verbosity)
      if verbosity:
         print ('Read Training Image from: \n%s' % (path))
         for j, img in enumerate(self.trainImages):
            print ('-'*60)
            print ('img            : %s' % self.trainFilenames[j])
            print ('image Size     : %d x %d'% (img.size[1], img.size[0]))
            print ('-'*60)
      self.trainImagesVectors = img_encoder.imagesToVectors(self.trainImages)
      #img_arr_temp = np.asarray(img.getdata(), dtype=np.uint8)
      #img_arr = img_arr_temp[:,:-1] 
      #self.input_bin = np.unpackbits(input, axis=0)

   def checkTagsTable(self, tags):
      for tag in tags:
         assert tag in self.tagsTable,\
         'We have not learned tag %s yet' % tag

   def changeTestingImg(self, path, verbosity=0):
      self.testImages, self.testFilenames, self.testTags = data_reader.getImagesAndTags(path)
      self.checkTagsTable(self.testTags)
      if verbosity:
         print ('Read Testing Image from: \n%s' % (path))
         for j, img in enumerate(self.testImages):
            print ('-'*60)
            print ('img            : %s' % self.testFilenames[j])
            print ('image Size     : %d x %d'% (img.size[1], img.size[0]))
            print ('-'*60)
      self.testImagesVectors = img_encoder.imagesToVectors(self.testImages)      

   def learnImages(self, verbosity=0):
      #activeArray = np.zeros(self.obj_module.sp.getNumColumns());
      for j,trainImage in enumerate(self.trainImagesVectors):
         self.obj_module.spatialPooling(trainImage, True)
      for j,trainImage in enumerate(self.trainImagesVectors):
         category = self.tagsTable.index(self.trainTags[j])
         activeArray = self.obj_module.spatialPooling(trainImage, False)
         self.clf.learn(activeArray, category)
         if verbosity:
            print 'Train Image : %r' % trainImage
            print 'ActiveList: %r' % activeList
   def checkAccuracy(self, datas, tags, verbosity=0):
      accuracy = 0.0
      if verbosity:
         print 'Check Accuracy, len = %s' % len(tags)
         print tags
      for j,data in enumerate(datas):
         category = self.tagsTable.index(tags[j])
         activeArray = self.obj_module.spatialPooling(data, False)
         inferred = self.clf.infer(activeArray)[0]
         if inferred == category:
            accuracy += 100.0/len(tags)
         if verbosity:
            print 'Pattern i = %s , answer = %s, inferred = %s' \
                  %(j, category, inferred) 
            print 'Check Result: %r' % (inferred==category)
      return accuracy

   def test(self):
      accuracy = self.checkAccuracy(self.testImagesVectors, self.testTags)
      #print self.testTags
      print 'testing accuracy: %s' % accuracy

   def train(self, minIteration=10, maxIteration=100, \
             minAccuracy=99.0, verbosity=0):
      accuracy = 0.0
      iteration = 0
      stop_threshold = 1.0
      while (minAccuracy > accuracy and \
            abs(minAccuracy - accuracy) > stop_threshold and  \
            iteration < maxIteration) or iteration < minIteration:
         iteration += 1
         self.clf.clear()
         self.learnImages()
         accuracy = self.checkAccuracy(self.trainImagesVectors, \
                                       self.trainTags)
         if verbosity:
            print 'Iteration = %s , Accuracy = %s ' % (iteration ,accuracy)

   def saveTestResult(self, mode, saveRoot, verbosity = 1):
      for tag in self.tagsTable:
         saveDir = saveRoot + '/' + tag
         if not os.path.exists(saveDir):
            os.mkdir(saveDir) 

      for j, image in enumerate(self.testImages):
         if mode in ('p', 'png'):
            fileExt = 'png'
            saveDir = saveRoot + '/' + self.testResultTag[j] 
            if not os.path.exists(saveDir):
              print 'Error: This tag %s is not included in tagsTable!' \
                     %(self.testResultTag[j])
            savePath = saveDir + '/' + \
                       self.testFilenames[j].split('/')[1].split('.')[0] +\
                       '.' + fileExt
            image.save(savePath)
         #elif mode in ('x', 'xml'):
