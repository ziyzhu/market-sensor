# Market Sensor
Foresee market behaviors based on Google News ([see my paper](https://github.com/zachzhu2016/market-sensor/blob/main/picture/marketsensor.pdf)).

This project aimed to implement a reusable pipeline for news & stock data and achieve high accuracy in predicting stock price movements with the results of sentiment analysis on related Google News articles. The baseline used to evaluate the prediction accuracies was random coin flipping. 

## Pipeline / Architecture
![pipeline](https://github.com/zachzhu2016/market-sensor/blob/main/picture/pipeline.png)
1.	Gathers data from both Google News API and Yahoo Finance API. (cached)
2.	Downloads article texts. (cached) 
3.	Adds sentiment value to downloaded articles by using the average of the sentiments of the sentences in article texts, sentences with less than 5 tokens / words were considered invalid and were skipped because sampling had shown that articles tend to have irrelevant data (e.g., phone number). (cached) 
4.	Fits the augmented article data into a DataFrame indexed by article URL’s. Stock data already sits in a DataFrame indexed by dates.
5.	Creates a timeline DataFrame based on the stock data, whose rows are indexed by dates, and append the URL’s of the articles published in a new column. 
6.	Computes price change percentage using stock’s opening and closing price. 
7.	Iterates through the timeline DataFrame and replaces empty sentiment values with sentiment values from the most recent day of the previous three days if that day had a non-empty sentiment value.
8.	Applies a scoring function and derives an overall score of an article based on their sentiment values. 
9.	Computes accuracies by comparing price change percentage with article scores. Predictions have three values, negative (sell), hold (no action), and positive (buy), and each of which’s accuracy is computed.  
10.	Lastly, graphs accuracies and covariances with stock groups and different window length (the number of days of Google News data, in the past, used to predict the stock price of the current day). 


## Example Figures
QCOM Example Figure:
![QCOM](https://github.com/zachzhu2016/market-sensor/blob/main/picture/QCOM_graph.png)

