import os, json, math, pyspark
from pyspark import SparkContext
from pyspark.sql import Row, SparkSession, DataFrame
from pyspark.sql.types import * 
from pyspark.sql.functions import col
from pyspark.mllib.stat import KernelDensity
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, startdate, enddate):
        self.startdate = startdate
        self.enddate = enddate


