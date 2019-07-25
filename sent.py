import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import json
from urllib.parse import quote

apiKey = ''
fromDate = '2019-07-05'

def getNews(query):
    endpoint = f'https://newsapi.org/v2/everything?language=en&pageSize=100&from={fromDate}&q={query}&apiKey={apiKey}'
    news = requests.get(endpoint)
    return news.json()

def getHeadlines(news):
    articles = news['articles']
    titles = [i['title'] for i in articles]
    descriptions = [i['description'] for i in articles]
    return titles + descriptions

def analyzeSentiments(headlines):
    sid = SentimentIntensityAnalyzer()
    sentiments = np.zeros(len(headlines))
    for sentence_idx, sentence in enumerate(headlines):
        try:
            ss = sid.polarity_scores(sentence)
            sentiments[sentence_idx] = ss['compound']
        except:
            pass
    return sentiments

query = 'BYND OR (beyond meat)'
encoded = quote(query)
news = getNews(encoded)
headlines = getHeadlines(news)
sentiments = analyzeSentiments(headlines)

print(sentiments.mean(), sentiments.std(), len(sentiments))
