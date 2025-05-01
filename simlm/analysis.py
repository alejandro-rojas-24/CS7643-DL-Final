import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 



def load_and_process_results(filepath):
    """Loads results from JSON Lines file and processes them into a DataFrame."""
    data = []
    print(f"Loading results from: {filepath}")

    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {i+1}")
                    continue 
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return pd.DataFrame()


    if not data:
        print("Error: No data loaded or all lines were malformed.")
        return pd.DataFrame()

    raw_result_df = pd.DataFrame(data)

    if "config" not in raw_result_df.columns:
        print("Error: 'config' column missing in the loaded data. Cannot process.")
        return pd.DataFrame()

    # Normalize the nested dictionary
    try:
        # Create a temporary series dropping rows where not a dict
        config_series = raw_result_df["config"].dropna()
        valid_indices = config_series[config_series.apply(isinstance, args=(dict,))].index
        if len(valid_indices) < len(raw_result_df):
             print(f"Warning: Dropping {len(raw_result_df) - len(valid_indices)} rows with invalid 'config' entries.")

        if not valid_indices.empty:
             config_df = pd.json_normalize(raw_result_df.loc[valid_indices, "config"], sep='_')
             config_df.index = valid_indices
        else:
             print("Warning: No valid 'config' entries found for normalization.")
             config_df = pd.DataFrame() 

        # Use only rows with valid indices
        result_df = pd.concat([raw_result_df.loc[valid_indices].drop(columns=["config"]), config_df], axis=1)

    except Exception as e:
        print(f"Error during JSON normalization of 'config': {e}")
        return pd.DataFrame()

    if result_df.empty:
        print("DataFrame is empty after processing 'config'.")
        return pd.DataFrame()

    result_df['error'] = pd.to_numeric(result_df['error'], errors='coerce') # Convert numeric errors, others become NaN

    if result_df.empty:
        print("Warning: No valid runs found after filtering NaNs in 'error'.")
        return pd.DataFrame()

    # Fill NaN iterations for CoT (effectively 1 iteration) or handle based on strategy
    if 'iterations_run' in result_df.columns:
         # Only fillna if the strategy is CoT? Or assume NaN means 1? Let's assume NaN means 1 for now.
         result_df['iterations_run'] = result_df['iterations_run'].fillna(1).astype(int)
    else:
         print("Warning: 'iterations_run' column missing.")
         result_df['iterations_run'] = 1 

    columns_to_keep = {
        "timestamp": "Timestamp",
        "success": "Success",
        "error": "Error",
        "time_taken": "Time Taken (s)",
        "iterations_run": "Iterations Run",
        "experiment_few_shot": "Num Few Shot",
        "final_h": "Final Height (m)",
        "final_v": "Final Velocity (m/s)",
        "actual_distance_bounce_3": f"Actual Distance Bounce 3 (m)", 
        "bounce_locations": "Bounce Locations (m)",
        "experiment_type": "Strategy",
        "experiment_target_distance": "Target Distance (m)",
        "experiment_max_iterations": "Max Iterations Allowed",
        "llm_service": "LLM Service",
        "llm_model_name": "LLM Model Name",
        "llm_temperature": "LLM Temperature",
        "ground_type": "Ground Type",
        "ground_difficulty": "Ground Difficulty",
    }

    if 'experiment_target_bounce_number' in result_df.columns:
         try:
             bounce_num = int(result_df['experiment_target_bounce_number'].iloc[0])
             columns_to_keep["actual_distance_bounce_3"] = f"Actual Distance Bounce {bounce_num} (m)"
         except (ValueError, TypeError, IndexError):
             print("Warning: Could not determine target bounce number. Using default column name.")

    # Select and rename columns, handling potential missing columns
    final_cols = {}
    missing_expected_cols = []
    for k, v in columns_to_keep.items():
        if k in result_df.columns:
            final_cols[k] = v
        else:
            if k != "experiment_few_shot": 
                missing_expected_cols.append(k)

    if missing_expected_cols:
        print(f"Warning: Expected columns not found in results: {missing_expected_cols}")

    # Filter DataFrame to only include columns that actually exist BEFORE renaming
    existing_cols_to_rename = {k: v for k, v in final_cols.items() if k in result_df.columns}
    result_df = result_df[list(existing_cols_to_rename.keys())].rename(columns=existing_cols_to_rename)

    if "Num Few Shot" in result_df.columns:
        # Ensure it's numeric, fill missing with 0, convert to integer
        result_df["Num Few Shot"] = pd.to_numeric(result_df["Num Few Shot"], errors='coerce').fillna(0).astype(int)
    else:
        print("Warning: 'experiment_few_shot' column not found after normalization. Adding 'Num Few Shot' column with default 0.")
        result_df["Num Few Shot"] = 0


    # Clean Categorical Values
    if "Strategy" in result_df.columns:
        result_df["Strategy"] = (
            result_df["Strategy"]
            .str.replace("baseline_cot", "Baseline CoT", regex=False)
            .str.replace("simlm", "SimLM", regex=False)
        )
    if "LLM Service" in result_df.columns:
        result_df["LLM Service"] = (
            result_df["LLM Service"].str.title().str.replace("Openai", "OpenAI", regex=False)
        )
    if "Ground Type" in result_df.columns:
        result_df["Ground Type"] = result_df["Ground Type"].str.title()
    else:
         print("Warning: 'Ground Type' column missing.")


    def get_condition(row):
        # Check required columns exist in the row index
        ground_type = row.get('Ground Type', 'Unknown')
        ground_difficulty = row.get('Ground Difficulty', None)

        if ground_type == 'Flat':
            return 'Flat Ground (Exp A)'
        elif ground_type in ('Uneven', 'Sinusoid'): 
             return 'Uneven Ground (Exp B)'
        elif ground_type == 'Interpolated':
            difficulty_str = f"{ground_difficulty:.1f}" if isinstance(ground_difficulty, (int, float)) else 'N/A'
            return f'Interpolated (Diff: {difficulty_str}) (Exp C)' 
        else:
            return ground_type 

    # Apply get_condition if required columns exist
    if 'Ground Type' in result_df.columns:
         result_df['Experiment Condition'] = result_df.apply(get_condition, axis=1)
    else:
         result_df['Experiment Condition'] = 'Unknown' 


    # Convert timestamp if exists
    if "Timestamp" in result_df.columns:
        result_df['Timestamp'] = pd.to_datetime(result_df['Timestamp'], errors='coerce')
        result_df = result_df.sort_values("Timestamp")
    else:
        print("Warning: 'Timestamp' column missing.")

    print(f"Processed {len(result_df)} results after cleaning.")
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
    plt.xticks(rotation=30, ha='right')
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
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(rect=[0, 0, 0.85 if hue_col else 1, 1]) 
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
        errorbar=('ci', 95) 
    )
    plt.title(title)
    plt.xlabel("Ground Difficulty (Interpolated)")
    plt.ylabel(metric_col.replace('_', ' ').title())
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.show()

def plot_iterations_histogram(df, title):
    """
    Plots a stacked histogram of iterations taken for SimLM runs,
    grouped by experiment condition.

    Args:
        df (pd.DataFrame): DataFrame with experiment results, requiring columns:
                           'Strategy', 'Iterations Run', 'Experiment Condition'.
                           Optionally 'Max Iterations Allowed'.
        title (str): The title for the plot.
    """
    simlm_df = df[df['Strategy'] == 'SimLM'].copy()

    if simlm_df.empty:
        print("No SimLM data found for iteration histogram.")
        return

    if 'Max Iterations Allowed' in df.columns:
         max_iter_allowed = pd.to_numeric(df['Max Iterations Allowed'].dropna().iloc[0], errors='coerce')
         if pd.isna(max_iter_allowed):
             max_iter_allowed = None 
    else:
         max_iter_allowed = None

    if max_iter_allowed is None and 'Iterations Run' in simlm_df.columns:
        max_iter_allowed = pd.to_numeric(simlm_df['Iterations Run'].dropna().max(), errors='coerce')
        if pd.isna(max_iter_allowed):
             max_iter_allowed = None 

    if max_iter_allowed is None:
        print("Warning: Could not determine Max Iterations Allowed. Defaulting to 5.")
        max_iter_allowed = 5
    else:
        max_iter_allowed = int(max_iter_allowed) 

    bins = np.arange(1, max_iter_allowed + 2) - 0.5

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    ax = sns.histplot(
        data=simlm_df,
        x='Iterations Run',
        hue='Experiment Condition', 
        multiple="stack",         
        bins=bins,
        palette='viridis',         
        shrink=0.8          
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Iterations Run by SimLM", fontsize=12)
    plt.ylabel("Number of Runs", fontsize=12)

    plt.xticks(np.arange(1, max_iter_allowed + 1))
    ax.tick_params(axis='both', which='major', labelsize=10)


    legend = ax.get_legend()

    if legend:
        legend.set_title('Experiment Condition')

        plt.setp(legend, bbox_to_anchor=(1.03, 1), loc='upper left')
    else:
        print("Warning: Seaborn did not automatically create a legend. Attempting manual creation.")
        try:
            plt.legend(title='Experiment Condition', bbox_to_anchor=(1.03, 1), loc='upper left')
        except Exception as e:
             print(f"Manual legend creation also failed: {e}")

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    plt.show()

def print_summary_stats(df):
    """Prints summary statistics grouped by strategy and condition with robust aggregation."""
    print("\n--- Summary Statistics ---")

    if df.empty:
        print("DataFrame is empty. Cannot calculate summary stats.")
        return

    numeric_cols = ['Error', 'Time Taken (s)', 'Iterations Run', 'Success'] 
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
             print(f"Warning: Column '{col}' expected for aggregation not found.")

    aggregations = {
        'count': ('Error', 'size'),
        'mean_error': ('Error', 'mean'),
        'median_error': ('Error', 'median'),
        'std_error': ('Error', 'std'),
        'success_rate_mean': ('Success', 'mean'), 
        'mean_time': ('Time Taken (s)', 'mean'),
    }

    def safe_simlm_iter_mean(x):
        strategies = df.loc[x.index, 'Strategy']
        simlm_iters = x[strategies == 'SimLM']
        simlm_iters_numeric = pd.to_numeric(simlm_iters, errors='coerce').dropna()
        if simlm_iters_numeric.empty:
            return np.nan 
        else:
            return simlm_iters_numeric.mean()

    if 'Iterations Run' in df.columns and 'Strategy' in df.columns:
         aggregations['mean_simlm_iters'] = ('Iterations Run', safe_simlm_iter_mean)

    try:
        summary = df.groupby(['Strategy', 'Experiment Condition'], observed=False, dropna=False).agg(
            **aggregations
        ).reset_index()
    except Exception as e:
        print(f"Error during aggregation: {e}")
        print("Problem might be due to non-numeric data in aggregated columns.")
        return 

    summary = summary.rename(columns={'success_rate_mean': 'success_rate'})
    summary['success_rate'] = (summary['success_rate'] * 100) # Calculate percentage

    cols_to_round_1 = ['success_rate', 'mean_simlm_iters']
    cols_to_round_2 = ['mean_error', 'median_error', 'std_error', 'mean_time']

    for col in cols_to_round_1:
        if col in summary.columns:
            summary[col] = summary[col].round(1)
    for col in cols_to_round_2:
         if col in summary.columns:
            summary[col] = summary[col].round(2)

    if 'mean_simlm_iters' in summary.columns:
        summary['mean_simlm_iters'] = summary['mean_simlm_iters'].fillna('N/A') # Or keep as NaN

    print(summary.to_string(index=False))
    print("-" * 50)

def plot_simlm_cot_error_ratio(df, model_col='LLM Model Name', strategy_col='Strategy', error_col='Error'):
    """Plots the ratio of SimLM error to COT error for different models."""
    mean_errors = df.groupby([model_col, strategy_col])[error_col].mean().unstack()

    mean_errors.dropna(inplace=True)

    print(mean_errors.head(5))

    #calc error ratio
    mean_errors['Error Ratio (SimLM/CoT)'] = mean_errors['SimLM'] / mean_errors['Baseline CoT']

    plt.figure(figsize=(10, 6))
    sns.barplot(x=mean_errors.index, y='Error Ratio (SimLM/CoT)', data=mean_errors)
    plt.title('Error Ratio of SimLM to CoT by Model')
    plt.xlabel('LLM Model')
    plt.ylabel('Error Ratio (SimLM/CoT)')
    plt.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

def plot_error_ratio_by_ground(df, model_col='LLM Model Name', strategy_col='Strategy', error_col='Error', ground_col='Ground Type'):
    """Plots the ratio of SimLM error to CoT error across different LLM models and ground types."""
    
    mean_errors = df.groupby([model_col, strategy_col, ground_col])[error_col].mean().unstack(level=strategy_col)
    
    expected_strategies = ['Baseline CoT', 'SimLM']
    for strategy in expected_strategies:
        if strategy not in mean_errors.columns:
            raise KeyError(f"Missing expected strategy: {strategy}. Found strategies: {mean_errors.columns.tolist()}")

    mean_errors = mean_errors.dropna()

    mean_errors['Error Ratio (SimLM/CoT)'] = mean_errors['SimLM'] / mean_errors['Baseline CoT']
    mean_errors = mean_errors.reset_index()

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=mean_errors, 
        x=model_col, 
        y='Error Ratio (SimLM/CoT)', 
        hue=ground_col, 
        palette='viridis'
    )
    plt.axhline(1, linestyle='--', color='red', label='Parity Line (Ratio = 1)')
    plt.title('Error Ratio (SimLM/CoT) by LLM Model and Ground Type')
    plt.xlabel('LLM Model')
    plt.ylabel('Error Ratio (SimLM / Baseline CoT)')
    plt.legend(title='Ground Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def summarize_average_error_by_ground(df, model_col='LLM Model Name', ground_col='Ground Type', error_col='Error'):
    """Creates a table of average error where rows are LLM models and columns are ground types."""
    summary = df.groupby([model_col, ground_col])[error_col].mean().unstack()
    summary = summary.dropna()
    summary = summary.round(2)
    print("\n--- Average Error by LLM Model and Ground Type ---")
    return summary

def summarize_improvement_by_ground(df, model_col='LLM Model Name', ground_col='Ground Type', strategy_col='Strategy', error_col='Error'):
    """Creates a table showing improvement (Baseline CoT error - SimLM error) by model and ground type."""
    mean_errors = df.groupby([model_col, ground_col, strategy_col])[error_col].mean().unstack()
    mean_errors['Improvement (m)'] = mean_errors['Baseline CoT'] - mean_errors['SimLM']
    summary = mean_errors['Improvement (m)'].unstack()
    summary = summary.dropna()
    summary = summary.round(2)
    print("\n--- Improvement (Baseline CoT - SimLM) by LLM Model and Ground Type ---")
    return summary

def summarize_relative_improvement(df, model_col='LLM Model Name', ground_col='Ground Type', strategy_col='Strategy', error_col='Error'):
    """Summarizes relative improvement percentage between Baseline CoT and SimLM."""
    mean_errors = df.groupby([model_col, ground_col, strategy_col])[error_col].mean().unstack()
    mean_errors['Relative Improvement (%)'] = (mean_errors['Baseline CoT'] - mean_errors['SimLM']) / mean_errors['Baseline CoT'] * 100
    summary = mean_errors['Relative Improvement (%)'].unstack()
    summary = summary.dropna()
    summary = summary.round(1)
    print("\n--- Relative Improvement (%) by LLM Model and Ground Type ---")
    return summary

def plot_error_vs_ground_difficulty(df):
    """Plots error as a function of ground difficulty for SimLM and CoT."""
    exp_c_df = df[df['Ground Type'] == 'Interpolated'].copy()
    if exp_c_df.empty:
        print("No data found for 'Interpolated' ground type.")
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=exp_c_df, x='Ground Difficulty', y='Error', hue='Strategy', marker='o')
    plt.title('Error vs Ground Difficulty')
    plt.xlabel('Ground Difficulty')
    plt.ylabel('Mean Error (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def summarize_failure_rate(df, error_threshold=20.0):
    """Summarizes the failure rate (errors above threshold) for each LLM and strategy."""
    df['Failure'] = df['Error'] > error_threshold
    failure_summary = df.groupby(['LLM Model Name', 'Strategy'])['Failure'].mean() * 100
    failure_summary = failure_summary.dropna()
    failure_summary = failure_summary.round(1)
    print("\n--- Failure Rate (%) by LLM Model and Strategy ---")
    return failure_summary

def plot_iterations_vs_error(df):
    """Plots the number of iterations vs final error for SimLM."""
    simlm_df = df[df['Strategy'] == 'SimLM']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=simlm_df, x='Iterations Run', y='Error')
    sns.regplot(data=simlm_df, x='Iterations Run', y='Error', scatter=False, color='red')
    plt.title('Iterations vs Final Error (SimLM)')
    plt.xlabel('Iterations Run')
    plt.ylabel('Final Error (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_histogram_error_distribution(df, strategy_col='Strategy', error_col='Error'):
    """ Histogram visual of error distribution by strategy"""
    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x=error_col, hue=strategy_col, kde=True, bins=30)
    plt.title('Error Distribution by Strategy')
    plt.xlabel('Error (m)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def perform_statistical_tests(df, metric_col='Error', group_col='Experiment Condition', strategy_col='Strategy', strategy_1='Baseline CoT', strategy_2='SimLM', alpha=0.05):
    """
    Performs independent t-tests to compare a metric between two strategies
    for each group defined by group_col.
    """

    print(f"\n--- Statistical Significance Tests ({strategy_1} vs {strategy_2} for {metric_col}) ---")
    if group_col not in df.columns or strategy_col not in df.columns or metric_col not in df.columns:
        print(f"Error: Required columns ('{group_col}', '{strategy_col}', '{metric_col}') not found.")
        return

    # Ensure strategies exist
    if strategy_1 not in df[strategy_col].unique() or strategy_2 not in df[strategy_col].unique():
         print(f"Warning: One or both strategies ('{strategy_1}', '{strategy_2}') not found in '{strategy_col}'. Skipping tests.")
         return

    unique_groups = df[group_col].unique()
    results = []

    for group in unique_groups:
        print(f"\nTesting for Group: {group}")
        group_df = df[df[group_col] == group]

        data_1 = group_df[group_df[strategy_col] == strategy_1][metric_col].dropna()
        data_2 = group_df[group_df[strategy_col] == strategy_2][metric_col].dropna()

        if len(data_1) < 2 or len(data_2) < 2:
            print(f"  Skipping: Insufficient data for one or both strategies (Need at least 2 data points).")
            results.append({'Group': group, 'Stat': np.nan, 'P-Value': np.nan, 'Significant': 'N/A (Insufficient Data)', 'N1': len(data_1), 'N2': len(data_2)})
            continue

        # Perform independent t-test (assumes unequal variance by default with Welch's t-test)
        t_stat, p_value = stats.ttest_ind(data_1, data_2, equal_var=False, nan_policy='omit')

        is_significant = p_value < alpha
        significance_str = f"Yes (p={p_value:.3g})" if is_significant else f"No (p={p_value:.3g})"

        print(f"  {strategy_1} Mean {metric_col}: {data_1.mean():.3f} (N={len(data_1)})")
        print(f"  {strategy_2} Mean {metric_col}: {data_2.mean():.3f} (N={len(data_2)})")
        print(f"  T-statistic: {t_stat:.3f}, P-value: {p_value:.3g}")
        print(f"  Difference significant at alpha={alpha}? {significance_str}")
        results.append({'Group': group, 'Stat': t_stat, 'P-Value': p_value, 'Significant': significance_str, 'N1': len(data_1), 'N2': len(data_2)})

    print("-" * 50)
    return pd.DataFrame(results)