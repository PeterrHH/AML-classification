import pandas as pd
import os

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


def save_5h_window_one_csv(
    input_csv: str,
    output_csv: str,
    window_start_hour: int = 10,
    window_length_hours: int = 5,
):
    df = pd.read_csv(input_csv)
    # 1) normalize
    df['Timestamp'] = df['Timestamp'] - df['Timestamp'].min()
    # 2) compute Hour
    df['Hour'] = (df['Timestamp'] // 3600).astype(int)
    # 3) filter
    lo = window_start_hour
    hi = window_start_hour + window_length_hours
    window_df = df[(df['Hour'] >= lo) & (df['Hour'] < hi)].copy()
    if window_df.empty:
        raise ValueError(f"No data in hours [{lo}, {hi})")

    # ensure directory exists
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    window_df.to_csv(output_csv, index=False)
    print(f"→ Wrote {len(window_df)} transactions (hours {lo}–{hi-1}) to {output_csv}")


# Example usage:
if __name__ == "__main__":
    # Adjust the input and output paths as necessary
    # save_filtered_days(
    #     input_csv="./data/formatted_transactions.csv",
    #     output_csv="./data/formatted_transactions_small.csv",
    #     days_to_keep=[3, 4, 5]
    # )
    # FOR QUICK TESTING
    save_5h_window_one_csv(
        input_csv="./data/formatted_transactions_small.csv",
        output_csv="./data/formatted_transactions_5_hrs.csv",
        window_start_hour = 10,
        window_length_hours = 5,
    )
