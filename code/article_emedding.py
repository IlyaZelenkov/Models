import torch
import json
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import pipeline
from process_article import Article
from rapidfuzz import process, fuzz
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util


class Embedding():

    def __init__(self, article: str, company: str) -> None:

        # Data variables:
        self.dependents: pd.DataFrame = pd.read_csv('data/dependents.csv', converters={"dependents": json.loads, "hurt_industries": json.loads})
        self.company_db: pd.DataFrame = pd.read_csv('data/article_stock.csv')[['company','ticker','industry','subindustry']].drop_duplicates(ignore_index=True)
        self.impact_db: pd.DataFrame = pd.read_csv('data/impact.csv')

        # Process Data: 
        self.article = Article(article)

        # Calculated weights: 
        self.sentiment: float = self.sentiment_analysis()
        self.impact: float = self.impact_analysis()
        self.relevance: float = self.relevance_analysis(company)

    def sentiment_analysis(self) -> float:
    
        ''' 

        Analyzes an article and returns a value between [-1,1]
        [-1]: Highly Negative article.
        [0]: Nuetral article.
        [1]: Highly Positive article. 

        Method: 
        Use some sentiment analyzer.. 

        '''

        sia: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()

        # Get sentiment scores
        sentiment_scores = sia.polarity_scores(self.article.processed_article)['compound']

        print(f'Sentiment Score: {sentiment_scores}')

        return sentiment_scores

    def impact_analysis(self) -> float:

        ''' 

        Analyzes an article and returns a value between [-1,1]
        [-1]: Highly negative stock impact.
        [0]: Does not affect stock.
        [1]: Highly Positive stock impact. 
        
        Method: 
        Cluster various events and observe their impact on stock markets. 
            - Catastrophic stock drop events: -1
            - No impact on stock price events, or data does not exist: 0
            - Amazing stock rise events: 1

        '''
        event_impacts: list = self.impact_db['event_type'].to_list()        

        print(f'Event classification of article: {self.article.get_event()[0]}')

        if self.article.get_event()[0] in event_impacts:


            return  self.impact_db.loc[self.impact_db['event_type'] == self.article.get_event()[0], 'average_impact'].iloc[0]

        else: 

            return 0
    
    def match_names(self, company: str) -> str:

        # Extract both the company name in the article and the list of available companies. 
        company = company.strip().upper()

        company_list = [str(company).strip().upper() for company in self.company_db['company'].tolist()]

        # Perform fuzzy matching
        company_matched = process.extractOne(company, company_list, scorer=fuzz.ratio)[0]

        return company_matched

    def relevance_analysis(self, company: str) -> float:

        
        company_article = self.match_names(self.article.get_company()[0])
        company_interest = self.match_names(company)

        print(f'Company in Article: {company_article}\nCompany from user input: {company_interest}')

        if company_article == company_interest: 

            return 1.00        

        else:

            company_article_df = self.company_db[self.company_db['company'] == company_article].set_index('company')
            company_interest_df = self.company_db[self.company_db['company'] == company_interest].set_index('company')

            if company_article_df['industry'].values[0] == company_interest_df['industry'].values[0]:

                return -1.00

            else:

                return 0.00

    def export_weights(self) -> list[float, float, float]:

        return [self.sentiment, self.impact, self.relevance]
    
