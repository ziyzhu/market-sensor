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

STOCKS_FILES = os.listdir(STOCKS_FOLDER)
FACTORS_FILES = os.listdir(FACTORS_FOLDER)

def confidence_interval(trials, f, nsamples, p):
    stats = sorted(map(lambda i: f(trials.sample(True, 1.0)), range(0, nsamples)))
    lower = int(nsamples * p / 2 - 1)
    upper = int(math.ceil(nsamples * (1 - p / 2)))
    return (stats[lower], stats[upper])

def fivepercent_VaR(trials):
    top_losses = trials.takeOrdered(max(int(trials.count() / 20), 1))
    return top_losses[-1]

def fivepercent_CVaR(trials):
    top_losses = trials.takeOrdered(max(int(trials.count() / 20), 1))
    return sum(top_losses) / len(top_losses)

def plot_distribution(samples):
    max_s = max(samples)
    min_s = min(samples)
    domain = np.arange(min_s, max_s, (max_s - min_s) / 100).tolist()
    kd = KernelDensity()
    samples_rdd = sc.parallelize(samples)
    kd.setSample(samples_rdd)
    densities = kd.estimate(domain)
    plt.plot(domain, densities, 'ro')
    plt.xlabel('Two Week Return')
    plt.ylabel('Density')
    plt.savefig('distribution.jpg')
    plt.show()

def featurize(row):
    squared = list(map(lambda x: math.copysign(1, x) * x * x, row))
    squarerooted = list(map(lambda x: math.copysign(1, x) * math.sqrt(abs(x)), row))
    feature = squared + squarerooted + row
    return feature

def linear_model(instrument, factor_features):
    lm = LinearRegression()
    return lm.fit(factor_features, instrument)

def compute_trial_returns(stock_returns, factor_returns, ntrials, parallelism, base_seed):
    factor_mat = np.transpose(factor_returns).tolist()
    factor_means = list(map(lambda factor: sum(factor) / len(factor), factor_returns))
    factor_features = list(map(featurize, factor_mat))
    linear_models = list(map(lambda stock: linear_model(stock, factor_features), stock_returns))
    factor_weights = list(map(lambda model: model.coef_.tolist(), linear_models))
    #binstruments = sc.broadcast(factor_weights)
    factor_cov = np.cov(factor_returns)

    seeds = range(base_seed, base_seed + parallelism)
    seed_rdd = sc.parallelize(seeds, parallelism)
    return seed_rdd.flatMap(lambda seed: trial_returns(seed, ntrials / parallelism, factor_weights, factor_means, factor_cov))

def trial_returns(seed, subntrials, instruments, factor_means, factor_cov):
    np.random.seed(seed)
    trial_returns = []
    for i in range(int(subntrials)):
        trial_factor_return = np.random.multivariate_normal(factor_means, factor_cov).tolist()
        trial_features = featurize(trial_factor_return)
        trial_returns.append(trial_return(trial_features, instruments))
    return trial_returns

def trial_return(trial_features, instruments):
    total_return = 0.0
    for instrument in instruments:
        total_return += instrument_trial_return(instrument, trial_features)
    return total_return / len(instruments)

def instrument_trial_return(instrument, trial_features):
    instrument_trial_return = 0.0
    for i in range(len(trial_features)):
        instrument_trial_return += trial_features[i] * instrument[i]
    return instrument_trial_return
    
def twoweek_returns(rows):
    mapped = []
    for i, row in enumerate(rows):
        if i + 10 <= len(rows):
            window = rows[i: i + 10]
            last = window[-1][1]
            first = window[0][1]
            if first != 0:
                mapped.append((last - first) / first)
            else:
                mapped.append(0)
    return mapped

def fillhistory(rows, start, end):
    crows = rows
    cdate = start
    filled = []

    while cdate < end: 
        if len(crows[1:]) > 0 and crows[1][0] == cdate:
            crows = crows[1:]
        filled.append((cdate, crows[0][1]))
        cdate += timedelta(days=1)
        if cdate.weekday() + 1 > 5:
            cdate += timedelta(days=2)

    return filled

def trim(rows, start, end):
    trimmed = list(filter(lambda row: row[0] >= start and row[0] <= end, rows))
    if trimmed and trimmed[0][0] != start:
        trimmed = [(start, trimmed[0][1])] + trimmed
    if trimmed and trimmed[-1][0] != end:
        trimmed = trimmed + [(end, trimmed[-1][1])]
    return trimmed

def parse(files, folder):
    parsed = []
    for fname in files:
        fpath = f'{folder}/{fname}'
        with open(fpath, 'r', encoding='utf-8-sig') as f:
            lines = [line.rstrip() for line in f]
            lines = lines[1:]
            rows = []
            for line in lines: 
                try:
                    row = line.split(',')
                    date = datetime.strptime(row[0], '%d-%b-%y')
                    openprice = float(row[1])
                    rows.append((date, openprice))
                except:
                    continue
            rows.reverse()
            parsed.append(rows) 
    return parsed

start = datetime(2009, 10, 23)
end = datetime(2014, 10, 23)

parsed_stocks = parse(STOCKS_FILES[:50], STOCKS_FOLDER)
parsed_factors = parse(FACTORS_FILES, FACTORS_FOLDER)

filtered_stocks = filter(lambda stock: len(stock) > 0, parsed_stocks)
filtered_factors = filter(lambda factor: len(factor) > 0, parsed_factors)

trimmed_stocks = list(map(lambda stock: trim(stock, start, end), filtered_stocks))
trimmed_factors = list(map(lambda factor: trim(factor, start, end), filtered_factors))

filled_stocks = list(map(lambda stock: fillhistory(stock, start, end), trimmed_stocks))
filled_factors = list(map(lambda factor: fillhistory(factor, start, end), trimmed_factors))

stock_returns = list(map(lambda stock: twoweek_returns(stock), filled_stocks))
factor_returns = list(map(lambda factor: twoweek_returns(factor), filled_factors))

#for i in range(3):
#    plot_distribution(factor_returns[i])

ntrials = 1000000
parallelism = 1000
base_seed = 1496
# trials = compute_trial_returns(stock_returns, factor_returns, ntrials, parallelism, base_seed)
# trials.cache()
factor_mat = np.transpose(factor_returns).tolist()
factor_means = list(map(lambda factor: sum(factor) / len(factor), factor_returns))
factor_features = list(map(featurize, factor_mat))
linear_models = list(map(lambda stock: linear_model(stock, factor_features), stock_returns))
factor_weights = list(map(lambda model: model.coef_.tolist(), linear_models))
# binstruments = sc.broadcast(factor_weights)
factor_cov = np.cov(factor_returns)

seeds = range(base_seed, base_seed + parallelism)
seed_rdd = sc.parallelize(seeds, parallelism)
trials = seed_rdd.flatMap(lambda seed: trial_returns(seed, ntrials / parallelism, factor_weights, factor_means, factor_cov))
trials.cache()
#trials_list = trials.collect()
#fivepercent_VaR(trials)
#fivepercent_CVaR(trials)
#plot_distribution(trials_list)

#VaR_conflevel = confidence_interval(trials, fivepercent_VaR, 100, 0.05)
#CVaR_conflevel = confidence_interval(trials, fivepercent_CVaR, 100, 0.05)


