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

    ''' analyze window's effect on covariance '''
    cov_sum = dict()
    windows = [1, 2, 3, 4, 5, 6, 7]
    for window in windows:
        covs = engine.analyze_cov(symbols, window=window, info='for all symbols', show_fig=False)
        cov_sum[window] = dict()
        for k in covs:
            cov_sum[window][k] = sum(covs[k])
    
    for k in ['price_text', 'price_title', 'title_text']: 
        plt.plot(windows, [cov_sum[window][k] for window in windows], label=k)

    plt.ylabel(f'Covariance Sum for {len(windows)} window days')
    plt.xlabel('Window in Days')
    plt.legend()
    plt.show()

    # analyze industry's effect on covariance
    # for tag in symbol_map:
    #     covs = engine.analyze_cov(symbol_map[tag], window=3, info=f'for {tag}')


