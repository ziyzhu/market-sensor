import numpy as np
import pandas as pd 
from instrument import * 
from article import *
from analysis import *

if __name__ == '__main__':
    startdate = datetime(2015, 1, 1)
    enddate = datetime(2017, 1, 1)
    interval = timedelta(weeks=4)

    engine = AnalyticEngine(startdate, enddate, interval)
    engine.prepare()
    
