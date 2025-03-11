import pandas as pd

# Define your thresholds: these represent your domain knowledge.
LOWER_BOUND = -20  # huge negative impact threshold
UPPER_BOUND = 20   # huge positive impact threshold

def custom_normalize(x, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    """
    Normalize x such that lower_bound maps to -1 and upper_bound maps to +1.
    Values below or above the bounds are clipped.
    """
    # Clip the value to the specified bounds
    if x < lower_bound:
        x = lower_bound
    elif x > upper_bound:
        x = upper_bound
    # Apply the normalization formula
    return 2 * (x - lower_bound) / (upper_bound - lower_bound) - 1

def main():
    # Read your JSON file into a DataFrame
    event_data: pd.DataFrame = pd.read_json('data/impact.json')
    
    event_types = []
    average_impacts = []
    
    # Process each column (event type)
    for col in event_data.columns:
        # Drop NaN values and flatten the lists
        data: pd.Series = event_data[col].dropna()
        flattened = data.explode().astype(float)
        
        # Compute the raw average impact for the event type
        raw_avg = flattened.mean()
        
        # Normalize using the custom thresholds
        norm_avg = custom_normalize(raw_avg)
        
        event_types.append(col)
        average_impacts.append(norm_avg)
    
    # Create a new DataFrame with the desired format
    output_df = pd.DataFrame({
        'event_type': event_types,
        'average_impact': average_impacts
    })
    
    # Write the new DataFrame to a CSV file
    output_df.to_csv('normalized.csv', index=False)
    print("CSV file 'normalized.csv' created successfully.")

if __name__ == '__main__':
    main()
