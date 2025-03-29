import json
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from rapidfuzz import fuzz, process
from process_question import QuestionProcessor
import torch

class Agent:

    def __init__(self) -> None:

        model_id = "mistralai/Mistral-7B-Instruct-v0.1"

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        self.prompt: str = input(f'Model: Please enter a question for the model to answer.\nUser Question: ')
        self.process_request()

    def process_request(self) -> None:

        self.prompt_params: QuestionProcessor = QuestionProcessor(self.prompt)
        self.impact: dict = pd.read_json('data/impact.json')

        self.companies: str | list = self.prompt_params.get_company()
        self.event: str = self.prompt_params.get_event()[0]
        self.scope: str = self.prompt_params.get_company_scope()
        intent: str = self.prompt_params.get_intent()

        self.validate_elements()

        match intent:

            case "general_info":
                self.get_general_info(multi=isinstance(self.companies, list))

            case "numeric_calc":
                self.calculate_request(multi=isinstance(self.companies, list))

            case "unknown":
                print("I wasn't able to understand your request!")

    def validate_elements(self) -> None:

        print("pre match", self.event, self.companies)

        # Normalize event keys and fuzzy match
        event_keys = list(self.impact.keys())
        event_key_map = {e.lower().replace(" ", "_"): e for e in event_keys}
        event_query = self.event.lower().replace(" ", "_")

        if event_query in event_key_map:
            
            self.event = event_key_map[event_query]
            
        else:
            
            best_event, score, _ = process.extractOne(event_query, event_key_map.keys(), scorer=fuzz.token_sort_ratio)
            
            if score >= 80:
                
                self.event = event_key_map[best_event]
                
            else:
                
                print(f"No confident match found for event '{self.event}'")

        if isinstance(self.companies, list):
            
            matched_companies = []
            
            for company in self.companies:
                
                comp_keys = list(self.impact[self.event].keys())
                comp_key_map = {c.lower(): c for c in comp_keys}
                query = company.lower()
                
                if query in comp_key_map:
                    
                    matched_companies.append(comp_key_map[query])
                    
                else:
                    
                    best_match, score, _ = process.extractOne(query, comp_key_map.keys(), scorer=fuzz.token_sort_ratio)
                    
                    if score >= 80:
                        
                        matched_companies.append(comp_key_map[best_match])
                        
                    else:
                        
                        print(f"⚠️ No confident match for '{company}' (best: {best_match}, score: {score})")
                        
            self.companies = matched_companies
            
        else:
            
            comp_keys = list(self.impact[self.event].keys())
            comp_key_map = {c.lower(): c for c in comp_keys}
            query = self.companies.lower()
            
            if query in comp_key_map:
                
                self.companies = comp_key_map[query]
                
            else:
                
                best_match, score, _ = process.extractOne(query, comp_key_map.keys(), scorer=fuzz.token_sort_ratio)
                
                if score >= 80:
                    
                    self.companies = comp_key_map[best_match]
                    
                else:
                    
                    print(f"⚠️ No confident match for '{self.companies}' (best: {best_match}, score: {score})")

        print("post match", self.event, self.companies)

    def get_general_info(self, multi=False) -> None:

        if multi:
            
            for company in self.companies:
                
                values = self.impact[self.event][company]
                summary = self.generate_summary_with_llm(company, self.event, values)
                print(f"\n\U0001F9FE Summary for {company}:\n{summary}\n")
                
        else:
            
            values = self.impact[self.event][self.companies]
            summary = self.generate_summary_with_llm(self.companies, self.event, values)
            print(f"\n\U0001F9FE Summary for {self.companies}:\n{summary}\n")

    def build_llm_prompt(self, company: str, event: str, summary_stats: dict) -> str:

        return f"""
    
                You are a financial analyst assistant.

                Here are the results of a statistical analysis of how the event \"{event}\" affected the company \"{company}\":

                - Number of articles analyzed: {summary_stats['count']}
                - Date range: {summary_stats['min_date']} to {summary_stats['max_date']}
                - Average total change: {summary_stats['avg_change']:.2%}
                - Pre-event average return: {summary_stats['avg_pre_mean']:.2%}
                - Post-event average return: {summary_stats['avg_post_mean']:.2%}
                - Average p-value: {summary_stats['avg_p_value']:.3f}
                - Overall direction: {summary_stats['direction']}

                Write an analytical and professional summary that explains the values that are included. Include as to what can be interprited from the result. 
                """

    def generate_summary_with_llm(self, company: str, event: str, records: list[dict]) -> str:

        total_changes = [r["total_change"] for r in records]
        pre_means = [r["pre_mean"] for r in records]
        post_means = [r["post_mean"] for r in records]
        p_values = [r["p_value"] for r in records]
        dates = [datetime.strptime(r["publish_date"], "%d/%m/%Y") for r in records]

        summary_stats = {
            "avg_change": np.mean(total_changes),
            "avg_pre_mean": np.mean(pre_means),
            "avg_post_mean": np.mean(post_means),
            "avg_p_value": np.mean(p_values),
            "direction": "positive" if np.mean(total_changes) > 0 else "negative",
            "count": len(records),
            "min_date": min(dates).strftime("%d %b %Y"),
            "max_date": max(dates).strftime("%d %b %Y"),
        }

        prompt = f"[INST] {self.build_llm_prompt(company, event, summary_stats)} [/INST]"

        result = self.generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=50256,)[0]['generated_text']

        return result[len(prompt):].strip()

    def calculate_request(self, multi=False) -> None:
        pass


def main():
    
    agent = Agent()


if __name__ == '__main__':
    main()
