import pandas as pd

def save_filtered_days(input_csv, output_csv, days_to_keep):
    """
    Filters the transactions to keep only specific days based on timestamp and saves to a new CSV.
    
    Parameters:
    - input_csv: path to the original AML transaction CSV
    - output_csv: path to save the filtered CSV
    - days_to_keep: list of days to retain (e.g., [3, 4, 5])
    """
    df = pd.read_csv(input_csv)
    min_timestamp = df['Timestamp'].min()
    df['Timestamp'] = df['Timestamp'] - min_timestamp  # Normalize to start at 0

    seconds_in_day = 24 * 3600
    filtered_df = pd.DataFrame()

    for day in days_to_keep:
        start = day * seconds_in_day
        end = (day + 1) * seconds_in_day
        day_df = df[(df['Timestamp'] >= start) & (df['Timestamp'] < end)]
        filtered_df = pd.concat([filtered_df, day_df], ignore_index=True)

    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered CSV saved to {output_csv} with {len(filtered_df)} transactions from days {days_to_keep}")

# Example usage:
if __name__ == "__main__":
    # Adjust the input and output paths as necessary
    save_filtered_days(
        input_csv="../formatted_transactions.csv",
        output_csv="../formatted_transactions_small.csv",
        days_to_keep=[3, 4, 5]
    )
