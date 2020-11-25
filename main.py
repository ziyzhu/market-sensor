import numpy as np
import pandas as pd 
from instrument import * 
from article import *
from analysis import *

if __name__ == '__main__':

    symbols = ['QCOM', 'TM', 'AAL', 'WKHS', 'BP', 'ASML', 'LUV', 'GM', 'DAL', 'TSLA', 'INTC', 'FCAU', 'FCAU', 'UAL', 'NVDA', 'CVX', 'XOM', 'JBLU', 'TOT']
    startdate = datetime(2015, 1, 1)
    enddate = datetime(2017, 1, 1)
    interval = timedelta(weeks=4)

    sc = SparkContext("local[*]", "AnalyticEngine")

    engine = AnalyticEngine(symbols, startdate, enddate, interval, sc)
    engine.prepare()
    engine.add_sentiment()
    article_df = engine.data['QCOM']['article_df']
    timeline_df = engine.data['QCOM']['timeline_df']

