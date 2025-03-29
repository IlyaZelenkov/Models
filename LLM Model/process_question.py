import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

class QuestionProcessor:

    def __init__(self, question: str):
        
        self.original_question = question.strip()
        self.processed_question = self.question_processing(question)
        self.attributes = self.extract_attributes()

    def question_processing(self, question: str) -> str:
        
        question = question.lower()
        question = re.sub(r'http\S+|www\S+|https\S+', '', question, flags=re.MULTILINE)
        question = re.sub(r'[^a-zA-Z0-9\s]', '', question)

        tokens = word_tokenize(question)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

        processed_question = ' '.join(tokens)
        print("Tokenized question:", processed_question)

        return processed_question

    def extract_attributes(self) -> dict:
        
        attributes: dict = {
            'company': None,
            'event': None,
            'intent': None,
            'company_scope': None,
        }

        # Load event labels
        event_labels: list = pd.read_csv('data/possible_events.csv')['event'].to_list()

        # Named Entity Recognition for company detection
        ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
        
        ner_results = ner_pipeline(self.original_question)
        companies = [entity['word'] for entity in ner_results if entity['entity_group'] == "ORG"]
        attributes['company'] = companies if companies else "Unknown"

        # Determine company scope
        question_lower = self.original_question.lower()
        
        if any(phrase in question_lower for phrase in 
        [
            "all companies", "which companies", "what companies", "every company",
            "most affected companies", "top companies"
        ]):
            
            attributes['company_scope'] = "all"
            
        elif len(companies) > 1:
            
            attributes['company_scope'] = "multiple"
            
        elif len(companies) == 1:
            
            attributes['company_scope'] = "single"
            
        else:
            
            if "what company" in question_lower or "which company" in question_lower:
                
                attributes['company_scope'] = "all"
                
            else:
                
                attributes['company_scope'] = "unknown"

        # Event classification
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        classification = classifier(self.processed_question, event_labels)
        attributes['events'] = classification['labels']
        attributes['scores'] = classification['scores']

        # Intent classification with descriptions
        intent_labels = [
            "general_info: Asking about who, what, or which companies are affected or what the general effects are of an event",
            "numeric_calc: Asking for a specific calculation like total change, average impact, or trends",
            "unknown: Cannot determine intent"
        ]
        intent_result = classifier(self.original_question, intent_labels)
        intent_label = intent_result['labels'][0].split(":")[0]  # extract raw label
        attributes['intent'] = intent_label

        return attributes

    def get_company(self) -> str:
        
        return self.attributes['company']

    def get_event(self) -> str:
        
        return self.attributes['events']

    def get_intent(self) -> str:
        
        return self.attributes['intent']

    def get_company_scope(self) -> str:
        
        return self.attributes['company_scope']

    def get_events_by_threshold(self, threshold: float) -> list:
        
        filtered_events = [
            event for event, score in zip(self.attributes['events'], self.attributes['scores']) 
            if score >= threshold
        ]
        
        return filtered_events

    def get_tokenized_question(self) -> str:
        
        return self.processed_question
