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
    # engine.load_all()

    article_df = engine.data['QCOM']['article_df']
    timeline_df = engine.data['QCOM']['timeline_df']
    article_dfs = [df_dict['article_df'] for df_dict in engine.data.values()]

    windows = list(range(31))
    engine.analyze_covs(symbols, windows=windows, save_fig=True, show_fig=False)
    engine.analyze_accuracies(symbols, windows=windows, save_fig=True, show_fig=False)
    # engine.analyze_accuracy(symbols, window=3, info='for all symbols', show_fig=True)
    # engine.analyze_accuracy(symbols, window=3, info='for all symbols', show_fig=True)


