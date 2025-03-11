import torch 
import numpy as np
import pandas as pd
import yfinance as yf
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from article_emedding import Embedding  # your custom embedding class

class PredictionModel(nn.Module):
    """
    This model calculates a stock prediction based on 9 inputs:
      - 3 embedding scores: sentiment, impact, relevance.
      - 1 volatility value.
      - 5 past trading day stock prices.
      
    The output is a range for the following 5 days:
      - output = [n1, n2, n3, n4, n5]

    """
    def __init__(self, input_dim=9, hidden_dim=16, output_dim=5):


        super(PredictionModel, self).__init__()

        # Input Layer.
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Activation function. 
        self.relu = nn.ReLU()

        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):

        # Forward pass input layer -> activation function -> hidden layer -> output layer. 

        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)

        return output

    
    @staticmethod
    def parse_date(date: str) -> datetime:

        for format in ('%Y-%m-%d', '%d/%m/%Y'):

            try:

                return datetime.strptime(date, format)

            except ValueError:

                continue

        raise ValueError(f"Date {date} is not in a recognized format.")

    
    def extract_volatility(self, ticker: str, ref_date: str = None) -> float:
        
        ''' Extract the volatility of the stocks for the past 6 months fro ARTICLE PUBLISHING DATE'''

        if ref_date:

            try:

                end_date_dt = self.parse_date(ref_date)

            except ValueError as e:

                print(e)

                return 0.0

        else:

            end_date_dt = datetime.today()

        end_date = end_date_dt.strftime('%Y-%m-%d')
        start_date_dt = end_date_dt - timedelta(days=182)
        start_date = start_date_dt.strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:

            return 0.0
        
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

        data['Daily_Return'] = data[price_col].pct_change()

        volatility = data['Daily_Return'].std()

        return volatility if not np.isnan(volatility) else 0.0

    
    def download_past_prices(self, ticker: str, ref_date: str, num_days: int = 5) -> np.ndarray:
        ''' Downloads the last 5 days of the data. '''

        try:

            ref_dt = self.parse_date(ref_date)

        except Exception as e:

            print(f"Failed to parse ref_date: {ref_date}")

            return None
        
        end_date = ref_dt.strftime('%Y-%m-%d')
        start_date = (ref_dt - timedelta(days=15)).strftime('%Y-%m-%d')
        data_past = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data_past.empty or len(data_past) < num_days:

            return None
        
        price_col = 'Adj Close' if 'Adj Close' in data_past.columns else 'Close'
        past_prices = data_past[price_col].values[-num_days:]

        return past_prices

    def download_future_prices(self, ticker: str, publish_date: str) -> np.ndarray:

        try:

            publish_dt = self.parse_date(publish_date)

        except Exception as e:

            print(f"Failed to parse publish_date: {publish_date}")
            return None
        
        start_date = (publish_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (publish_dt + timedelta(days=15)).strftime('%Y-%m-%d')

        data_future = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data_future.empty or len(data_future) < 5:

            return None
        
        price_col = 'Adj Close' if 'Adj Close' in data_future.columns else 'Close'

        return data_future[price_col].values[:5]

    def train_model(self, df: pd.DataFrame, num_epochs: int = 10, learning_rate: float = 0.001, checkpoint_path: str = None):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):

            epoch_loss = 0.0
            valid_records = 0

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            for idx, row in df.iterrows():

                article_text = row['snippet']
                company = row['company']
                ticker = row['ticker']
                publish_date = row['date']
                
                # Get embedding scores and convert to floats.
                emb = Embedding(article_text, company)
                embedding_scores = [float(x) for x in emb.export_weights()]  # [sentiment, impact, relevance]
                
                # Compute volatility.
                company_volatility = float(self.extract_volatility(ticker, ref_date=publish_date))
                
                # Download past 5 days prices and force to floats.
                past_prices = self.download_past_prices(ticker, ref_date=publish_date)

                if past_prices is None or len(past_prices) < 5:

                    continue

                # Fix inhomogeneous shape by extracting first element if necessary.
                past_prices = [float(x[0]) if isinstance(x, list) else float(x) for x in past_prices.tolist()]
                
                # Build input vector.
                input_list = embedding_scores + [company_volatility] + past_prices

                if len(input_list) != 9:

                    continue

                input_features = np.array(input_list, dtype=np.float32)
                input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
                
                # Determine target prices.
                target_prices = None

                if pd.isnull(row['post_20_v']):

                    target_prices = self.download_future_prices(ticker, publish_date)

                else:

                    try:

                        post_20_v = eval(row['post_20_v'])
                        target_prices = np.array(post_20_v).flatten()[:5]

                    except Exception as e:

                        target_prices = None
                
                if target_prices is None or len(target_prices) < 5:

                    continue

                target_prices = [float(x[0]) if isinstance(x, list) else float(x) for x in target_prices.tolist()]
                target_tensor = torch.tensor(target_prices, dtype=torch.float32).unsqueeze(0)
                
                # Print iteration details.
                print(f"Record {idx}:")
                print(f'')
                print(f"  Past 5 Days Prices: {past_prices}")
                print(f"  Actual Next 5 Days Prices: {target_prices}")
                
                # Forward pass.
                output = self.forward(input_tensor)
                print(f"  Predicted Next 5 Days Prices: {output.detach().numpy().flatten().tolist()}")
                
                loss = criterion(output, target_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_records += 1

                if checkpoint_path is not None:

                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }

                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            if valid_records > 0:

                avg_loss = epoch_loss / valid_records

                print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

            else:

                print(f"Epoch [{epoch+1}/{num_epochs}] No valid records processed.")

            
 

    def predict(self, article_text: str, company: str, ticker: str) -> np.ndarray:

        emb = Embedding(article_text, company)
        embedding_scores = [float(x) for x in emb.export_weights()]
        company_volatility = float(self.extract_volatility(ticker))
        past_prices = self.download_past_prices(ticker, ref_date=datetime.today().strftime('%Y-%m-%d'))

        if past_prices is None or len(past_prices) < 5:

            raise ValueError("Not enough past price data for prediction.")
        
        past_prices = [float(x[0]) if isinstance(x, list) else float(x) for x in past_prices.tolist()]

        input_list = embedding_scores + [company_volatility] + past_prices
        input_features = np.array(input_list, dtype=np.float32)
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():

            predicted = self.forward(input_tensor).detach().numpy()

        return predicted

    def load_checkpoint(self, checkpoint_path: str, optimizer=None):

        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Parameter checkpoint loaded from {checkpoint_path}")

    def save_checkpoint(self, optimizer, checkpoint_path: str):

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(checkpoint, checkpoint_path)

        print(f"Parameter checkpoint saved to {checkpoint_path}")

def main():

    data = pd.read_csv('data/article_stock.csv')

    model = PredictionModel()

    checkpoint_path = 'model_parameters/checkpoint.pth'
    
    try:

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.load_checkpoint(checkpoint_path, optimizer=optimizer)
        print("Resuming training from checkpoint...")

    except Exception as e:

        print("No checkpoint found or error loading checkpoint. Starting fresh training.")
        optimizer = None
    
    model.train_model(data, num_epochs=10, learning_rate=0.001, checkpoint_path=checkpoint_path)
    
    test_article = "Your news article text here"
    test_company = "Your Company Name"
    test_ticker = "TICKER"  # Replace with the actual ticker symbol.
    
    predicted_prices = model.predict(test_article, test_company, test_ticker)
    print("Predicted 5-day closing prices:", predicted_prices)

if __name__ == '__main__':

    main()
