#!/usr/bin/env python

from obj_recog_test import object_recognition_testbench as obj_testbench
from parameter import parameter
import sys, getopt

def main(argv):
   p = parameter()
   try:
      opts, args = getopt.getopt(argv, "hl:t:s:d:p:",\
                   ["help", "learn=", "test=", "saveMode=",\
                   "dimension=", "precision="])
   except getopt.GetoptError:
      print 'run.py -l <trainSet> -t <testSet> -d <nodeDimension> \
            -p <precision>'
      print 'run.py -h for more help'
      sys.exit()
   saveEnable = False
   for opt, arg in opts:
      if opt in ('-h', '--help'):
         print '-l(--learn) <trainSet> -t(--test) <testSet>'
         sys.exit()
      #elif opt in ('-l', '--learn'):
      #   self.trainingDataset = self.dataPath + dataset
      #   print 'Set Train Set : %s' % dataset
      #elif opt in ('-t', '--test'):
      #   self.testingDataset = self.dataPath + dataset
      elif opt in ('-s', '--saveMode'):
         saveEnable = True
         saveMode = arg
      elif opt in ('-d', '-p' , '--dimension', '--precision'):
         p.setTestParameter(opt, arg)
      else:
         p.setSet(opt, arg)
   obj_tb = obj_testbench( columnDimensions = (p.nodeDim, p.nodeDim), \
                           precision = p.precision)
   obj_tb.changeTrainingImg(p.trainingDataset)
   obj_tb.changeTestingImg(p.testingDataset)
   obj_tb.train(30, 100,99.0, 0)
   obj_tb.test()
   print 'NodeDimension: %i' %p.nodeDim
   print '-'*20
   print
   if saveEnable:
      obj_tb.saveTestResult(saveMode, p.savePath)

if __name__ == "__main__":
   main(sys.argv[1:])
