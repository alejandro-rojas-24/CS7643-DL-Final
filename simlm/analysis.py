from scipy import stats 
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import matplotlib.ticker as mtick 

def map_ground_to_condition(row):
    ground_type = row.get('Ground Type', 'Unknown')
    ground_difficulty = row.get('Ground Difficulty', None) # Might be NaN or missing
    if ground_type == 'Flat': return 'Flat Ground (Exp A)'
    if ground_type in ('Uneven', 'Sinusoid'): return 'Uneven Ground (Exp B)'
    if ground_type == 'Interpolated':
        if pd.notna(ground_difficulty) and isinstance(ground_difficulty, (int, float)): return f'Interpolated (Diff: {ground_difficulty:.1f}) (Exp C)'
        else: return 'Interpolated (Exp C)' # Fallback
    if isinstance(ground_type, str): return ground_type.title()
    return 'Unknown'

def load_and_process_results(filepath):
    data = []
    print(f"Loading results from: {filepath}")
    if not os.path.exists(filepath): print(f"Error: File not found at {filepath}"); return pd.DataFrame()
    try: 
        with open(filepath, 'r') as f:
            for i, line in enumerate(f): data.append(json.loads(line))
    except: print("Error reading/parsing file"); return pd.DataFrame()
    if not data: print("No data loaded"); return pd.DataFrame()
    raw_result_df = pd.DataFrame(data)
    if "config" not in raw_result_df.columns: print("Error: 'config' column missing."); return pd.DataFrame()

    normalized_configs = []
    valid_indices = []
    for index, row in raw_result_df.iterrows():
        config_data = row.get('config')
        if isinstance(config_data, dict):
            normalized = pd.json_normalize(config_data, sep='_')
            normalized_configs.append(normalized)
            valid_indices.append(index)
        else: pass 

    if not valid_indices: return pd.DataFrame()
    config_df = pd.concat(normalized_configs); config_df.index = valid_indices
    result_df = pd.concat([raw_result_df.loc[valid_indices].drop(columns=["config"]), config_df], axis=1)
    if result_df.empty: return pd.DataFrame()

    columns_to_keep = { "timestamp": "Timestamp", "success": "Success_Raw", "error": "Error_Raw", "time_taken": "Time Taken (s)", "iterations_run": "Iterations Run", "experiment_few_shot": "Num Few Shot", "final_h": "Final Height (m)", "final_v": "Final Velocity (m/s)", "actual_distance_bounce_3": f"Actual Distance Bounce 3 (m)", "bounce_locations": "Bounce Locations (m)", "experiment_type": "Strategy", "experiment_target_distance": "Target Distance (m)", "experiment_max_iterations": "Max Iterations Allowed", "llm_service": "LLM Service", "llm_model_name": "LLM Model Name", "llm_temperature": "LLM Temperature", "ground_type": "Ground Type", "ground_difficulty": "Ground Difficulty", "experiment_target_bounce_number": "Target Bounce Number", "experiment_tolerance": "Tolerance (m)" }
    target_bounce_num = 3; columns_to_keep["actual_distance_bounce_3"] = f"Actual Distance Bounce {target_bounce_num} (m)"
    rename_map = {k: v for k, v in columns_to_keep.items() if k in result_df.columns}
    result_df = result_df[list(rename_map.keys())].rename(columns=rename_map)
    if "Error_Raw" in result_df.columns: result_df['Error'] = pd.to_numeric(result_df['Error_Raw'], errors='coerce')
    else: result_df['Error'] = np.nan
    if "Iterations Run" in result_df.columns: result_df['Iterations Run'] = pd.to_numeric(result_df['Iterations Run'], errors='coerce').fillna(1).astype(int)
    else: result_df['Iterations Run'] = 1
    if "Num Few Shot" in result_df.columns: result_df["Num Few Shot"] = pd.to_numeric(result_df["Num Few Shot"], errors='coerce').fillna(0).astype(int)
    else: result_df["Num Few Shot"] = 0
    if 'Error' in result_df.columns and 'Tolerance (m)' in result_df.columns:
         result_df['Tolerance (m)'] = pd.to_numeric(result_df['Tolerance (m)'], errors='coerce')
         result_df['Success'] = (result_df['Error'].notna() & result_df['Tolerance (m)'].notna() & (result_df['Error'] <= result_df['Tolerance (m)']))
    elif 'Success_Raw' in result_df.columns: result_df['Success'] = result_df['Success_Raw'].fillna(False).astype(bool)
    else: result_df['Success'] = False
    result_df['Success'] = result_df['Success'].astype(bool)
    if "Strategy" in result_df.columns: result_df["Strategy"] = result_df["Strategy"].astype(str).str.replace("baseline_cot", "Baseline CoT", regex=False).str.replace("simlm", "SimLM", regex=False)
    if "LLM Service" in result_df.columns: result_df["LLM Service"] = result_df["LLM Service"].astype(str).str.title().str.replace("Openai", "OpenAI", regex=False)
    if "Ground Type" in result_df.columns: result_df["Ground Type"] = result_df["Ground Type"].astype(str).str.title()
    if 'Ground Type' in result_df.columns: result_df['Experiment Condition'] = result_df.apply(map_ground_to_condition, axis=1)
    else: result_df['Experiment Condition'] = 'Unknown'
    if "Timestamp" in result_df.columns: result_df['Timestamp'] = pd.to_datetime(result_df['Timestamp'], errors='coerce'); result_df = result_df.sort_values("Timestamp").reset_index(drop=True)
    result_df = result_df.drop(columns=['Error_Raw', 'Success_Raw'], errors='ignore')
    print(f"Processed {len(result_df)} results after cleaning.")
    return result_df


def summarize_fewshot_performance(df, group_by_cols=None):
    if group_by_cols is None: group_by_cols = ['Experiment Condition', 'Strategy']
    required_cols = group_by_cols + ['Num Few Shot', 'Error', 'Success']
    if not all(col in df.columns for col in required_cols): print(f"Error: Missing required columns for summary: {[c for c in required_cols if c not in df.columns]}"); return pd.DataFrame()
    df_filtered = df.dropna(subset=['Error']).copy();
    if df_filtered.empty: print("Warning: No rows with valid 'Error' found for summary."); return pd.DataFrame()
    aggregation = {'Error': ['mean', 'median', 'std', 'count'],'Success': ['mean']}
    try:
        summary_df = df_filtered.groupby(group_by_cols + ['Num Few Shot']).agg(aggregation)
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        summary_df = summary_df.rename(columns={'Error_mean': 'Mean Error','Error_median': 'Median Error','Error_std': 'Std Dev Error','Error_count': 'Valid Run Count','Success_mean': 'Success Rate'}).reset_index()
        return summary_df
    except Exception as e: print(f"Error during aggregation for summary: {e}"); return pd.DataFrame()



def plot_metric_by_fewshot_single_hist(df, metric='Error', title=None,
                                  shots_to_compare=None,
                                  group_hue='Experiment Condition', 
                                  x_group='Strategy', 
                                  palette='colorblind'):
    """
    Plots a specified metric against few-shot count on a single axes,
    grouping bars by specified categories.

    Args:
        df (pd.DataFrame): Processed results DataFrame.
        metric (str): The column name of the metric to plot on the y-axis ('Error' or 'Success').
        title (str, optional): Plot title. If None, a default title is generated.
        shots_to_compare (list, optional): List of few-shot counts to include. Defaults to all.
        group_hue (str): Column name for color grouping (e.g., 'Experiment Condition').
        x_group (str): Column name for grouping bars side-by-side within each x-tick
                       (e.g., 'Strategy'). Set to None to not group on x.
        palette (str or dict): Color palette for seaborn.
    """
    if metric not in ['Error', 'Success']:
        print(f"Error: Metric '{metric}' not supported by this plot function. Use 'Error' or 'Success'.")
        return
    required_cols = ['Num Few Shot', metric, group_hue]
    if x_group: required_cols.append(x_group)
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns for single plot: {missing}")
        return

    plot_df = df.copy()
    if shots_to_compare:
        plot_df = plot_df[plot_df['Num Few Shot'].isin(shots_to_compare)]
    else:
        shots_to_compare = sorted(plot_df['Num Few Shot'].unique())

    if plot_df.empty: print(f"No data found for the specified few-shot counts: {shots_to_compare}."); return

    # Ensure categorical types for grouping columns
    plot_df[group_hue] = plot_df[group_hue].astype(str)
    if x_group: plot_df[x_group] = plot_df[x_group].astype(str)


    # Metric specific handling
    y_label = metric
    plot_metric = metric
    if metric == 'Error':
        plot_df = plot_df.dropna(subset=['Error'])
        y_label = "Mean Absolute Error (m)"
    elif metric == 'Success':
        y_label = "Success Rate"
        plot_df[plot_metric] = plot_df[metric].astype(float)


    if plot_df.empty: print(f"No valid data remains for metric '{metric}' after filtering."); return

    hue_order = sorted(plot_df[group_hue].unique())
    x_group_order = sorted(plot_df[x_group].unique()) if x_group else None


    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7)) 
    ax = sns.barplot(
        data=plot_df,
        x='Num Few Shot',
        y=plot_metric,
        hue=group_hue,
        order=sorted(shots_to_compare),
        hue_order=hue_order, 
        palette=palette,
        errorbar=('ci', 95) 
    )

    if title is None:
        title = f"{y_label} vs. Num Few-Shot Examples by {group_hue}"
        if x_group: title += f" and {x_group}"
    plt.title(title, fontsize=15)
    plt.xlabel("Number of Few-Shot Examples", fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Adjust legend position
    plt.legend(title=group_hue, bbox_to_anchor=(1.03, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    plt.show()



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
    mean_errors = df.groupby([model_col, strategy_col])[error_col].mean().unstack()

    mean_errors.dropna(inplace=True)

    print(mean_errors.head(5))

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