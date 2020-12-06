from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkContext
from pyspark.ml import Pipeline
# from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import LinearSVC
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string 
import spacy
nlp = spacy.load("en_core_web_sm")

sc = SparkContext("local[*]", "Price Prediction")
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("ERROR")

parser = English()
stopwords = set(STOP_WORDS)
punctuations = set(string.punctuation)

def tokenize(sentence):
    tokens = parser(sentence)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    tokens = [word for word in tokens if word not in stopwords and word not in punctuations and word != '']
    return tokens

history = engine.histories[0]
group = history.groups[0]
articles = group.search['articles']
rdd = spark.sparkContext.parallelize([article.get('text', '') for article in articles])
rdd = rdd.map(tokenize)
tf = hashingTF.transform(rdd)
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

for article in articles:
    pass

