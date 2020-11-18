import numpy as np
import pandas as pd 
from stock import * 
from article import *

if __name__ == '__main__':

    START_DATE = datetime(2015, 1, 1)
    END_DATE = datetime(2017, 1, 1)
    INTERVAL = timedelta(weeks=4)

    # instruments = load_instruments(START_DATE, END_DATE, readcache=False, writecache=True)
    instruments = load_instruments(START_DATE, END_DATE, readcache=True, writecache=False)

    dfs = []
    article_dfs = []
    histories = []
    for instrument in instruments:
        df = instrument.df.copy()
        startdate, enddate = instrument.date_range()
        if not all([startdate, enddate]):
            # continue
            pass

        # history = load_history(instrument, startdate, enddate, INTERVAL, readcache=False, writecache=True)
        history = load_history(instrument, startdate, enddate, INTERVAL, readcache=True, writecache=False)
        histories.append(history)

        articles = history.get_aligned_articles()
        articles_dict = dict()
        columns = ['published', 'title', 'link', 'source', 'id']
        for col in columns:
            articles_dict[col] = []
            for article in articles:
                articles_dict[col].append(article[col])

        article_df = pd.DataFrame.from_dict(articles_dict).set_index('link', drop=False, verify_integrity=True)
        article_series = article_df.groupby('published')['link'].apply(list).rename('links')
        merged_df = pd.merge(instrument.df, article_series, left_index=True, right_index=True, how='left')

        df = merged_df[['Open', 'links']].replace({np.nan: None})

        df.name = instrument.id
        article_df.name = instrument.id
        dfs.append(df)
        article_dfs.append(article_df)

        # df.to_csv(f'./data/{instrument.id}_df.csv')
        # article_df.to_csv(f'./data/{instrument.id}_article_df.csv')
    
    for history in histories:
        history.load_text()
        history.cache()

    

