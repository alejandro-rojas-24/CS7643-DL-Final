import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_process_results(filepath):
    data = []
    print(f"Loading results from: {filepath}")
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            data.append(json.loads(line))

    if not data:
        print("Error: No data loaded.")
        return pd.DataFrame()

    raw_result_df = pd.DataFrame(data)

    # Normalize the nested 'config' dictionary
    config_df = pd.json_normalize(raw_result_df["config"], sep='_') 

    # Combine results and config, drop raw config column
    result_df = pd.concat([raw_result_df.drop(columns=["config"]), config_df], axis=1)

    # Data Cleaning and Feature Engineering
    # Filter out runs where target bounce wasn't reached (error is NaN)
    # We might want to analyze these failures separately later
    result_df = result_df[result_df["error"].notna()].copy()
    if result_df.empty:
        print("Warning: No valid runs found after filtering NaNs in 'error'.")
        return pd.DataFrame()

    # Fill NaN iterations for CoT (effectively 1 iteration)
    result_df['iterations_run'] = result_df['iterations_run'].fillna(1).astype(int)

    # Define standard columns to keep
    columns_to_keep = {
        "timestamp": "Timestamp",
        "success": "Success",
        "error": "Error",
        "time_taken": "Time Taken (s)",
        "iterations_run": "Iterations Run",
        "final_h": "Final Height (m)",
        "final_v": "Final Velocity (m/s)",
        "actual_distance_bounce_3": f"Actual Distance Bounce {int(result_df['experiment_target_bounce_number'].iloc[0])} (m)",
        "bounce_locations": "Bounce Locations (m)",
        "experiment_type": "Strategy",
        "experiment_target_distance": "Target Distance (m)",
        "experiment_max_iterations": "Max Iterations Allowed",
        "llm_service": "LLM Service",
        "llm_model_name": "LLM Model Name",
        "llm_temperature": "LLM Temperature",
        "ground_type": "Ground Type",
        "ground_difficulty": "Ground Difficulty", 
        # "ground_amplitude": "Ground Amplitude",
        # "ground_frequency": "Ground Frequency",
    }

    # Select and rename columns, handling potential missing columns gracefully
    final_cols = {}
    for k, v in columns_to_keep.items():
        if k in result_df.columns:
            final_cols[k] = v
        else:
            print(f"Warning: Expected column '{k}' not found in results.")

    result_df = result_df[list(final_cols.keys())].rename(columns=final_cols)

    # Clean Categorical Values 
    result_df["Strategy"] = (
        result_df["Strategy"]
        .str.replace("baseline_cot", "Baseline CoT", regex=False)
        .str.replace("simlm", "SimLM", regex=False)
    )
    result_df["LLM Service"] = (
        result_df["LLM Service"].str.title().str.replace("Openai", "OpenAI", regex=False)
    )
    result_df["Ground Type"] = result_df["Ground Type"].str.title()

    def get_condition(row):
        if row['Ground Type'] == 'Flat':
            return 'Flat Ground'
        elif row['Ground Type'] == 'Uneven' or row['Ground Type'] == 'Sinusoid': # Handle variations
             return 'Uneven Ground (Exp B)'
        elif row['Ground Type'] == 'Interpolated':
            difficulty = row['Ground Difficulty']
            return f'Interpolated (Diff: {difficulty:.1f})' 
        else:
            return row['Ground Type'] 

    if 'Ground Difficulty' in result_df.columns:
         result_df['Experiment Condition'] = result_df.apply(get_condition, axis=1)
    else:
         result_df['Experiment Condition'] = result_df['Ground Type'] # Fallback if difficulty missing

    # Convert timestamp
    result_df['Timestamp'] = pd.to_datetime(result_df['Timestamp'])
    result_df = result_df.sort_values("Timestamp")

    print(f"Processed {len(result_df)} valid results.")
    return result_df


def plot_metric_distribution(df, metric_col, group_col, hue_col, title, yscale='linear', showfliers=True):
    plt.figure(figsize=(12, 7))
    order = sorted(df[group_col].unique()) if group_col else None
    hue_order = sorted(df[hue_col].unique()) if hue_col else None
    sns.boxplot(
        data=df,
        x=group_col,
        y=metric_col,
        hue=hue_col,
        order=order,
        hue_order=hue_order,
        showfliers=showfliers
    )
    plt.yscale(yscale)
    plt.title(title)
    plt.xlabel(group_col.replace('_', ' ').title())
    plt.ylabel(metric_col.replace('_', ' ').title())
    if hue_col:
        plt.legend(title=hue_col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout(rect=[0, 0, 0.85 if hue_col else 1, 1])
    plt.show()

def plot_success_rate(df, group_col, hue_col, title):
    success_data = df.groupby([group_col, hue_col] if hue_col else [group_col], observed=False)['Success'].mean().reset_index()
    success_data['Success Rate (%)'] = success_data['Success'] * 100

    plt.figure(figsize=(12, 7))
    order = sorted(df[group_col].unique()) if group_col else None
    hue_order = sorted(df[hue_col].unique()) if hue_col else None
    sns.barplot(
        data=success_data,
        x=group_col,
        y='Success Rate (%)',
        hue=hue_col,
        order=order,
        hue_order=hue_order,
        palette="viridis"
    )
    plt.title(title)
    plt.xlabel(group_col.replace('_', ' ').title())
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 105) 
    if hue_col:
       plt.legend(title=hue_col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout(rect=[0, 0, 0.85 if hue_col else 1, 1]) # Adjust layout for legend
    plt.show()

def plot_metric_by_difficulty(df, metric_col, title):
    """Plots a metric against ground difficulty for Experiment C."""
    exp_c_df = df[df['Ground Type'] == 'Interpolated'].copy()
    if exp_c_df.empty:
        print("No data found for 'Interpolated' ground type to plot by difficulty.")
        return
    if 'Ground Difficulty' not in exp_c_df.columns:
        print("Error: 'Ground Difficulty' column not found for difficulty plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=exp_c_df,
        x='Ground Difficulty',
        y=metric_col,
        hue='Strategy',
        marker='o',
        errorbar=('ci', 95) # Show confidence interval
    )
    plt.title(title)
    plt.xlabel("Ground Difficulty (Interpolated)")
    plt.ylabel(metric_col.replace('_', ' ').title())
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.show()

def plot_iterations_histogram(df, title):
    """Plots a histogram of iterations taken for SimLM runs."""
    simlm_df = df[df['Strategy'] == 'SimLM'].copy()
    if simlm_df.empty:
        print("No SimLM data found for iteration histogram.")
        return

    max_iter_allowed = df['Max Iterations Allowed'].iloc[0] # Assumes constant
    bins = np.arange(1, max_iter_allowed + 2) - 0.5 # Center bins on integers

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=simlm_df,
        x='Iterations Run',
        hue='Experiment Condition', # Color by condition
        multiple="stack", # Stack bars for different conditions
        bins=bins,
        discrete=True,
        shrink=0.8 # Add space between bars
    )
    plt.title(title)
    plt.xlabel("Iterations Run by SimLM")
    plt.ylabel("Number of Runs")
    plt.xticks(np.arange(1, max_iter_allowed + 1)) # Ensure integer ticks
    plt.legend(title='Experiment Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.show()

def print_summary_stats(df):
    """Prints summary statistics grouped by strategy and condition with robust aggregation."""
    print("\n--- Summary Statistics ---")

    if df.empty:
        print("DataFrame is empty. Cannot calculate summary stats.")
        return

    # --- Data Type Check (Debugging Aid) ---
    # print("Data types before aggregation:\n", df.dtypes)
    # print("Unique values in 'Error':", df['Error'].unique()[:10]) # Check for non-numeric
    # print("Unique values in 'Time Taken (s)':", df['Time Taken (s)'].unique()[:10])
    # print("Unique values in 'Iterations Run':", df['Iterations Run'].unique()[:10])
    # -----------------------------------------

    # Ensure relevant columns are numeric, coercing errors to NaN
    numeric_cols = ['Error', 'Time Taken (s)', 'Iterations Run', 'Success'] # Success is bool ~ int
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
             print(f"Warning: Column '{col}' expected for aggregation not found.")

    # Define aggregations
    aggregations = {
        'count': ('Error', 'size'), # Count based on a column expected to be present
        'mean_error': ('Error', 'mean'),
        'median_error': ('Error', 'median'),
        'std_error': ('Error', 'std'),
        'success_rate_mean': ('Success', 'mean'), # Calculate mean of boolean/numeric Success
        'mean_time': ('Time Taken (s)', 'mean'),
    }

    # Define the safer lambda function for SimLM iterations separately
    def safe_simlm_iter_mean(x):
        # x is the 'Iterations Run' series for the current group
        # First, get the 'Strategy' for the corresponding indices in the original df
        strategies = df.loc[x.index, 'Strategy']
        # Filter the 'Iterations Run' series for only SimLM runs
        simlm_iters = x[strategies == 'SimLM']
        # Ensure it's numeric and drop NaNs introduced by coerce or filtering
        simlm_iters_numeric = pd.to_numeric(simlm_iters, errors='coerce').dropna()
        if simlm_iters_numeric.empty:
            return np.nan # Return NaN if no valid SimLM runs in this group
        else:
            return simlm_iters_numeric.mean()

    # Only add SimLM calculation if the necessary columns exist
    if 'Iterations Run' in df.columns and 'Strategy' in df.columns:
         aggregations['mean_simlm_iters'] = ('Iterations Run', safe_simlm_iter_mean)

    # Perform the aggregation
    try:
        summary = df.groupby(['Strategy', 'Experiment Condition'], observed=False, dropna=False).agg(
            **aggregations # Use dictionary unpacking
        ).reset_index()
    except Exception as e:
        print(f"Error during aggregation: {e}")
        print("Problem might be due to non-numeric data in aggregated columns.")
        # Optionally print problematic group data here if possible
        return # Exit if aggregation fails

    # --- Post-processing and Formatting ---
    # Rename success rate column
    summary = summary.rename(columns={'success_rate_mean': 'success_rate'})
    summary['success_rate'] = (summary['success_rate'] * 100) # Calculate percentage

    # Rounding (apply only to columns that exist)
    cols_to_round_1 = ['success_rate', 'mean_simlm_iters']
    cols_to_round_2 = ['mean_error', 'median_error', 'std_error', 'mean_time']

    for col in cols_to_round_1:
        if col in summary.columns:
            summary[col] = summary[col].round(1)
    for col in cols_to_round_2:
         if col in summary.columns:
            summary[col] = summary[col].round(2)

    # Handle potential NaN in mean_simlm_iters after rounding if needed
    if 'mean_simlm_iters' in summary.columns:
        summary['mean_simlm_iters'] = summary['mean_simlm_iters'].fillna('N/A') # Or keep as NaN

    print(summary.to_string(index=False))
    print("-" * 50)