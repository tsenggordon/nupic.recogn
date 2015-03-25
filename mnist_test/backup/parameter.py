import os

class parameter():
   
   def __init__(self):
      self.trainingDataset = '../data_set/OCR/characters/hex_all.xml'
      self.testingDataset = '../data_set/OCR/characters/hex_normal.xml'
      self.minAccuracy = 200.0
      self.xTrainingCycles = 5
      self.dataPath = '../data_set/OCR/characters/'
      self.savePath = '../data_set/OCR/characters/Result'
      self.datasetPrefix = 'hex'
      self.datasetExt = 'xml'
      self.nodeDim = 32
      self.precision = 8
      if not os.path.exists(self.savePath):
         os.mkdir(self.savePath)
   def setSet(self, opt, arg):
      dataset = ''
      setname = ''
      if arg in ('a', 'all'):
         setname = 'all'
      elif arg in ('n', 'normal'):
         setname = 'normal'
      elif arg in ('i', 'italic'):
         setname = 'italic'
      elif arg in ('b', 'bold'):
         setname = 'bold'
      else:
         setname = 'all'
      dataset = self.datasetPrefix + '_' + setname + '.' + self.datasetExt
      if opt in ('-l', '--learn'):
         self.trainingDataset = self.dataPath + dataset
         print 'Set Train Set : %s' % dataset
      elif opt in ('-t', '--test'):
         self.testingDataset = self.dataPath + dataset
         print 'Set Test Set : %s' % dataset

   def setTestParameter(self, opt, arg):
      if opt in ('-d', '--dimension'):
         self.nodeDim = int(arg)
         print 'Set Spatial Pooler Node Dimension = %s' % arg
      else:
         self.precision = int(arg)
         print 'Set Connection Perminence Precision = %s' %arg
