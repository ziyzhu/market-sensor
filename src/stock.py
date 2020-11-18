from tqdm import tqdm
from time import sleep
from datetime import datetime
import cache
import json
import pandas
import yfinance as yf

TICKERS_PATH = '../data/tickers.json' 

class Instrument:
    def __init__(self, symbol, shortname=None, info=None):
        self.id = symbol
        self.symbol = symbol
        self.shortname = shortname
        self.info = info
        self.df = None

    def date_range(self):
        if len(self.df) > 0:
            startdate = self.df.index.to_pydatetime()[0]
            enddate = self.df.index.to_pydatetime()[-1]
            return startdate, enddate
        return None, None
    
    def qstr(self):
        return self.shortname

    @staticmethod
    def readcache(symbol):
        instrument = Instrument(symbol)
        d = cache.readcache(f'{Instrument.__name__}_{instrument.id}')
        instrument.__dict__.update(d)
        return instrument
    
    def cache(self):
        cache.writecache(f'{self.__class__.__name__}_{self.id}', self.__dict__)

    def __repr__(self):
        return f'Instrument(id={self.id}, symbol={self.symbol}, shortname={self.shortname})'

def get_symbols():
    symbols = []
    with open(TICKERS_PATH) as f:
        tkdict = json.load(f)
        symbols = [symbol for symbols in tkdict.values() for symbol in symbols]
    return symbols

def load_instruments(startdate, enddate, readcache=True, writecache=False):

    instruments = []
   
    if readcache: 
        query = f'{Instrument.__name__}'
        for objname in cache.listcache(query):
            symbol = objname.split('_')[1]
            instrument = Instrument.readcache(symbol)
            instruments.append(instrument)
        return instruments

    symbols = get_symbols()
    tickers = yf.Tickers(' '.join(symbols))

    infos = []
    for ticker in tqdm(tickers.tickers):
        while True:
            try:
                infos.append(ticker.info)
                break
            except:
                sleep(2)

    histories = tickers.download(group_by='ticker',\
                                 start='2010-11-10',\
                                 end='2020-11-10',\
                                 auto_adjust=True,\
                                 threads=True)
    
    for info in infos: 
        instrument = Instrument(symbol=info.get('symbol'), shortname=info.get('shortName'), info=info)
        start = startdate.strftime('%Y-%m-%d')
        end = enddate.strftime('%Y-%m-%d')
        instrument.df = histories[instrument.symbol][start:end].dropna()
        instruments.append(instrument)
        if writecache:
            instrument.cache()

    return instruments


