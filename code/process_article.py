import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

class Article:

    def __init__(self, article: str):

        # Keep the original article for tasks sensitive to case for company recognition.
        self.original_article = article.strip()

        # Processed version for other tasks (e.g. event classification)
        self.processed_article = self.article_processing(article)
        self.attributes = self.extract_attributes()

    def article_processing(self, article: str) -> str:

        # Lowercase the text and remove URLs and punctuation, etc.
        article = article.lower()
        article = re.sub(r'http\S+|www\S+|https\S+', '', article, flags=re.MULTILINE)
        article = re.sub(r'[^a-zA-Z0-9\s]', '', article)
        
        tokens = word_tokenize(article)

        stop_words = set(stopwords.words('english'))

        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

        processed_article = ' '.join(tokens)

        print("Tokenized article:", processed_article)

        return processed_article
    
    def extract_attributes(self) -> dict:

        attributes: dict = {'company': None, 'event': None}
        
        # Load event labels from CSV.
        event_labels: list = pd.read_csv('data/possible_events.csv')['event'].to_list()

        # Use the original article for NER to retain capitalization and punctuation.
        ner_pipeline = pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english", 
            aggregation_strategy="simple"
        )
        ner_results = ner_pipeline(self.original_article)

        # Look for organization entities (ORG) and collect them.
        companies = [entity['word'] for entity in ner_results if entity['entity_group'] == "ORG"]

        attributes['company'] = companies if companies else "Unknown"

        # Classify the Event with Zero-shot Classification.
        classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli"
        )
        classification = classifier(self.processed_article, event_labels)
        attributes['events'] = classification['labels']
        attributes['scores'] = classification['scores']

        return attributes  # Ensure the attributes dictionary is returned

    def get_company(self) -> str:

        return self.attributes['company']

    def get_event(self) -> str:

        return self.attributes['events']

    def get_events_by_threshold(self, threshold: float) -> list:

        # Filter out the events with a confidence score below the threshold.
        filtered_events = []

        for event, score in zip(self.attributes['events'], self.attributes['scores']):

            # Only add events that meet or exceed the threshold.
            if score >= threshold:

                filtered_events.append(event)

        return filtered_events

    def get_tokenized_article(self) -> str:

        return self.processed_article