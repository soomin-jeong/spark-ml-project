#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pyspark import SparkContext, SparkConf
from main import ArrivalDelayMachineLearningRunner
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

class Launcher(object):

    def showUsage(self):
        print("ERROR- BAD SYNTAX")
        print("Usage: launcher.py dataset_absolute_path model_chosen")
        print("Model chosen can take values: dt, rf, lr")
        sys.exit(1)

    def checkSyntax(self):
        conf = dict()
        print("check syntax")
        if len(sys.argv) != 3:
            self.showUsage()
        else:
            if str(sys.argv[2]) == "dt":
                conf.update({"model":"dt"})
            elif str(sys.argv[2]) == "rf":
                conf.update({"model":"rf"})
            elif str(sys.argv[2]) == "lr":
                conf.update({"model":"lr"})
            else:
                self.showUsage()
            conf.update({"data":sys.argv[2]})
        return conf

    def launchApp(self, conf):
        print("launching")
        runner = ArrivalDelayMachineLearningRunner()
        runner.run(conf)


if __name__ == '__main__':
    print("in process")
    # Testing purposes:
    conf = {}
    conf.update({"data":"2007.csv.bz2"})
    conf.update({"model":"dt"})
    #conf = self.checkSyntax() without testing remove comment
    launcher = Launcher()
    launcher.launchApp(conf)
