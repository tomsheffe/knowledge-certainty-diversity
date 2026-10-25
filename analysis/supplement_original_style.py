"""Original submitted S6/S7 visual designs, with finalized analysis inputs.
Plot functions preserve the original heatmap and switching-panel layouts.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_and_plot_confidence_heatmap(exp3_human_subjects, exp4_human_llm_post_process_data, bot_bot_bot_data_parsed):
    """
    Processes four distinct datasets to compare confidence and accuracy,
    and visualizes the results as a styled heatmap.

    Args:
        exp3_human_subjects (pd.DataFrame): Data for Human - Internal Medicine.
                                         Requires 'conf_val_start' and 'ans_start_correct' columns.
        exp4_human_llm_post_process_data (pd.DataFrame): Data for Human & LLM - Pediatrics.
                                                  Requires 'doc.init_question_confidence',
                                                  'doc.is_correct_first_answer',
                                                  'doc.answer_confidence_level',
                                                  and 'bot_before_correct' columns.
        bot_bot_bot_data_parsed (pd.DataFrame): Data for LLM - General Medicine.
                                                Requires 'conf_val_start' and
                                                'ans_start_correct' columns.
    """

    def create_confidence_accuracy_and_count_series(df, confidence_col, accuracy_col):
        """
        Rounds confidence values to the nearest integer (0–5) and computes:
          - mean accuracy for each confidence level
          - sample count for each confidence level.
        Returns a tuple (mean_series, count_series) with index 0–5.
        """
        df = df.copy()
        # Round confidence to nearest integer.
        df["confidence_int"] = pd.to_numeric(df[confidence_col], errors="coerce").round()
        # Keep only levels 0 to 5.
        df = df[df["confidence_int"].isin(range(0, 6))]
        # Compute mean accuracy and count for each confidence level.
        mean_series = df.groupby("confidence_int")[accuracy_col].mean()
        count_series = df.groupby("confidence_int")[accuracy_col].count()
        # Ensure all levels 0-5 appear.
        mean_series = mean_series.reindex(range(0, 6))
        count_series = count_series.reindex(range(0, 6))
        return mean_series, count_series

    # --- 1. Data Processing ---

    # Create series for each dataset.
    s_exp3_human_mean, s_exp3_human_count = create_confidence_accuracy_and_count_series(
        exp3_human_subjects, confidence_col="conf_val_start", accuracy_col="ans_start_correct"
    )

    s_exp4_human_llm_human_mean, s_exp4_human_llm_human_count = create_confidence_accuracy_and_count_series(
        exp4_human_llm_post_process_data, confidence_col="doc.init_question_confidence", accuracy_col="doc.is_correct_first_answer"
    )

    s_exp4_human_llm_llm_mean, s_exp4_human_llm_llm_count = create_confidence_accuracy_and_count_series(
        exp4_human_llm_post_process_data, confidence_col="doc.answer_confidence_level", accuracy_col="bot_before_correct"
    )

    s_botbot_mean, s_botbot_count = create_confidence_accuracy_and_count_series(
        bot_bot_bot_data_parsed, confidence_col="conf_val_start", accuracy_col="ans_start_correct"
    )

    # Combine the four series into one DataFrame.
    comparison_table = pd.DataFrame({
        "Humans \n Internal Medicine \n (Experiment 3) \n (chance level: 50%)": s_exp3_human_mean,
        "Humans \n  Pediatrics \n (Experiment 4) \n (chance level: 25%)": s_exp4_human_llm_human_mean,
        "LLM \n Pediatrics \n (Experiment 4) \n (chance level: 25%)": s_exp4_human_llm_llm_mean,
        "LLM \n General Medicine \n (Experiments 1+2) \n (chance level: 10–25%)": s_botbot_mean
    }, index=range(0, 6))

    # Create a corresponding table for counts.
    count_table = pd.DataFrame({
       "Humans \n Internal Medicine \n (Experiment 3) \n (chance level: 50%)": s_exp3_human_count,
       "Humans \n  Pediatrics \n (Experiment 4) \n (chance level: 25%)": s_exp4_human_llm_human_count,
       "LLM \n Pediatrics \n (Experiment 4) \n (chance level: 25%)": s_exp4_human_llm_llm_count,
       "LLM \n General Medicine \n (Experiments 1+2) \n (chance level: 10–25%)": s_botbot_count
    }, index=range(0, 6))

    # Prepare annotation strings for each cell: "mean\n(N=count)".
    annot_table = comparison_table.copy().astype(str)
    for level in comparison_table.index:
        for col in comparison_table.columns:
            mean_val = comparison_table.loc[level, col]
            count_val = count_table.loc[level, col]
            if pd.isnull(mean_val):
                annot_table.loc[level, col] = ""
            else:
                annot_table.loc[level, col] = f"{mean_val*100:.0f}%\n(N={int(count_val)})"

    # Reverse the order so that confidence 5 is at the top.
    comparison_table = comparison_table.iloc[::-1]
    annot_table = annot_table.iloc[::-1]

    # --- 2. Plotting ---

    # Create a custom colormap: red (low), yellow (mid), green (high).
    custom_cmap = LinearSegmentedColormap.from_list('submitted_s6',
        ['#b70d00', '#e34a33', '#fdae61', '#a6d96a', '#1a9850'], N=256)
    custom_cmap.set_bad("#bdc3c7") # Set color for NaN values

    # Plot the heatmap with improved aesthetics.
    plt.figure(figsize=(10, 7.5))
    ax = sns.heatmap(
        comparison_table,
        annot=annot_table,
        fmt="",
        cmap=custom_cmap,
        vmin=0,
        vmax=1,
        cbar=True,
        linewidths=1.5,
        linecolor='black',
        annot_kws={'color': 'black', 'size': 15} # Annotation styling
    )

    # Customize titles and labels
    plt.title("Mean Accuracy by Confidence Level", fontsize=24, fontweight='bold', pad=30)
    plt.ylabel("Confidence Level", fontsize=14, fontweight='bold', color='black')
    
    # Bold the tick labels
    plt.xticks(rotation=0, ha='center', fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold', color='black')

    ax.set_xticklabels(
    ax.get_xticklabels(),
    fontsize=10,            # size
    fontweight='bold',
    color='black'        # any matplotlib-compatible colour
)
    
    # Move x-axis labels to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    return plt.gcf()
    
    # Optionally print the final data table
    print("Comparison Table (Accuracy):")
    print(comparison_table.to_string(float_format="%.2f"))

def fmt_p(p_value):
    """
    Formats a p-value for clean display on the plot.
    Rounds to 3 decimal places or shows <0.001.
    """
    if p_value < 0.001:
        return "<0.001"
    else:
        return f"{p_value:.3f}"

def plot_ame_pro(df, sort_by='AME_pp', title_text=None):
    """
    Generate a professional, publication-quality error-bar plot of
    average marginal effects.

    Args:
        df (pd.DataFrame): DataFrame containing the AME results. Must have columns:
                           'Condition', 'AME_pp', 'CI_low_pp', 'CI_high_pp', 'p'.
        sort_by (str): The column name to sort the conditions by.
                       Defaults to 'AME_pp'.
        title_text (str): The main title for the plot. If None, a default
                          title is used.

    Key Improvements:
    - Sorts effects for easy comparison.
    - Uses a clean, modern plot style ('seaborn-v0_8-whitegrid').
    - Removes chart junk (unnecessary top/right borders).
    - Uses smarter, dynamic text placement for p-values.
    - Employs a clear title/subtitle hierarchy.
    - Built on Matplotlib's object-oriented API for better control.
    """
    # 1. --- Data Preparation & Sorting ---
    # Sort the dataframe by the effect size for clear visual hierarchy
    # A .copy() is used to avoid SettingWithCopyWarning
    # Retain the submitted figure's condition order as well as its design.
    df_plot = df.set_index('Condition').loc[
        ['LLM-trio', 'Human-duo', 'Human-trio', 'LLM-duo']].reset_index()
   
    # 2. --- Plotting Setup ---
    # Use a professional-looking style
    plt.style.use('seaborn-v0_8-whitegrid')
   
    # Create figure and axes using the object-oriented API
    # Adjust figsize for better aspect ratio
    fig, ax = plt.subplots(figsize=(7, 8))
   
    # y_pos is an array of positions for the bars [0, 1, 2, ...]
    y_pos = np.arange(len(df_plot))

    # Calculate error values for the error bar. This must be a 2xN array
    # where the first row is the lower error and the second is the upper error.
    x_errors = [
        df_plot["AME_pp"] - df_plot["CI_low_pp"],
        df_plot["CI_high_pp"] - df_plot["AME_pp"]
    ]

    # 3. --- Plotting the Core Data ---
    # Plot the main points and confidence intervals
    ax.errorbar(
        x=df_plot["AME_pp"],
        y=y_pos,
        xerr=x_errors,
        fmt='o',             # Use 'o' for circle markers
        color='black',       # A neutral, strong color for the main data
        capsize=5,           # Size of the caps on the error bars
        elinewidth=1.5,      # Line width of the CIs
        markeredgecolor='black',
        markerfacecolor='gray', # Fill color for the markers
        markersize=8,        # Size of the markers
        zorder=2             # Plot data on top of the grid and vline
    )

    # 4. --- Annotations and Text ---
    # Add p-value annotations with smarter, dynamic positioning
    for i, row in enumerate(df_plot.itertuples()):
        # Format the p-value string for display
        if fmt_p(row.p) == "<0.001":
            p_tag = "p<0.001" # Use a more compact format
        else:
            p_tag = f"p = {fmt_p(row.p)}"
           
        # Dynamically calculate offset based on plot's x-axis range
        # This makes the placement robust to different data scales
        axis_width = ax.get_xlim()[1] - ax.get_xlim()[0]
        offset = axis_width * 0.015  # 1.5% of the axis width
       
        # Position text just to the right of the high CI bar
        text_x_pos = row.CI_high_pp + offset
       
        ax.text(
            x=text_x_pos,
            y=y_pos[i],
            s=p_tag,
            verticalalignment="center",
            horizontalalignment="left",
            fontsize=11,
            color='dimgray' # Use a softer color for annotations
        )

    # 5. --- Final Styling and Labels ---
    # Add a vertical line at x=0 for a reference point
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

    # Set y-axis ticks and labels from the sorted dataframe
    ax.set_xticks([-20, -15, -10, -5, 0])
    ax.tick_params(axis='x', labelsize=16)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Condition"], fontsize=12)
   
    # Remove y-tick marks for a cleaner look along the axis
    ax.tick_params(axis='y', length=0)

    # Set labels and titles with clear hierarchy
    ax.set_xlabel("Avg. Marginal Effect of Confidence (percentage points)", fontsize=12, labelpad=15)
   
    # Use a main title (suptitle) and a descriptive subtitle (ax.title)
    if title_text is None:
        title_text = "How a 1-unit ↑ in Confidence changes P(switch)"
    fig.suptitle(title_text, fontsize=16, fontweight='bold', ha='center')
    ax.set_title("Average Marginal Effects with 95% Confidence Intervals", fontsize=12, pad=20, color='gray')

    # De-clutter by removing unnecessary plot borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Lighten the remaining spines so they are less prominent
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')

    # Adjust layout to prevent titles from overlapping
    # rect=[0, 0, 1, 0.96] makes space at the top for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.96])
   
    return fig

def figure_s6():
    from common import load_exp3_human, load_exp4_human_llm, load_bot_bot_trio
    _, human3 = load_exp3_human(0.62)
    from pathlib import Path
    pairs = pd.read_csv(Path(__file__).resolve().parents[1] / 'derived_data' /
                        's6_pediatrics_calibration.csv').rename(columns={
        'human_confidence': 'doc.init_question_confidence',
        'human_correct': 'doc.is_correct_first_answer',
        'model_confidence': 'doc.answer_confidence_level',
        'model_correct': 'bot_before_correct'})
    _, llm = load_bot_bot_trio()
    with plt.rc_context():
        plt.rcdefaults()
        return create_and_plot_confidence_heatmap(human3, pairs, llm)


def figure_s7():
    from fig_s7_switching import run
    _, _, ame = run()
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update({'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
        return plot_ame_pro(ame)
