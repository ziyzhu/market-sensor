import numpy as np
import pandas as pd 
from instrument import * 
from article import *
from analysis import *
from stats import * 

if __name__ == '__main__':

    symbol_map = {'semiconductor': ['QCOM', 'ASML', 'INTC', 'NVDA'],\
                  'automobile': ['TM', 'WKHS', 'GM', 'TSLA', 'FCAU', 'CVX'],\
                  'oil': ['BP', 'XOM', 'TOT'],\
                  'airline': ['AAL', 'LUV', 'DAL', 'UAL', 'JBLU']}
    symbols = [symbol for symbol_list in symbol_map.values() for symbol in symbol_list]

    startdate = datetime(2015, 1, 1)
    enddate = datetime(2017, 1, 1)
    interval = timedelta(weeks=4)

    engine = AnalyticEngine(symbol_map, startdate, enddate, interval)
    engine.add_all()

    article_df = engine.data['QCOM']['article_df']
    timeline_df = engine.data['QCOM']['timeline_df']
    article_dfs = [df_dict['article_df'] for df_dict in engine.data.values()]

    windows = list(range(1, 32))
    res_a = engine.analyze_accuracies(windows=windows, save_fig=True, show_fig=False)
    res_c = engine.analyze_covs(windows=windows, save_fig=True, show_fig=False)


