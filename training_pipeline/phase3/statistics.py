"""
Phase 3: Statistics & Visualization

Đọc result.csv và tạo thống kê, visualization.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def load_csv(csv_file_path: str = None) -> pd.DataFrame:
    """
    Đọc file CSV.
    
    Args:
        csv_file_path: Đường dẫn đến CSV file
    
    Returns:
        DataFrame
    """
    if csv_file_path is None:
        csv_file_path = config.RESULT_CSV_PATH
    
    csv_file = Path(csv_file_path)
    
    if not csv_file.exists():
        print(f"CSV file not found: {csv_file_path}")
        return None
    
    return pd.read_csv(csv_file)


def print_statistics(df: pd.DataFrame):
    """
    In thống kê mô tả.
    
    Args:
        df: DataFrame
    """
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    metrics = {
        "Latency (ms)": ["m1_latency", "m2_latency", "m3_latency"],
        "Total Input Tokens": ["m1_total_input_token", "m2_total_input_token", "m3_total_input_token"],
        "Total Output Tokens": ["m1_total_output_token", "m2_total_output_token", "m3_total_output_token"],
        "K": ["m1_k", "m2_k", "m3_k"],
        "Judge Score": ["m1_judge_score", "m2_judge_score", "m3_judge_score"]
    }
    
    for metric_name, cols in metrics.items():
        print(f"\n{metric_name}:")
        print("-" * 60)
        stats_df = df[cols].describe()
        stats_df.columns = ["Method 1 (Baseline)", "Method 2 (Adaptive Mini)", "Method 3 (Adaptive N5)"]
        print(stats_df)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print("\nAverage Latency (ms):")
    print(f"  Method 1: {df['m1_latency'].mean():.2f}")
    print(f"  Method 2: {df['m2_latency'].mean():.2f}")
    print(f"  Method 3: {df['m3_latency'].mean():.2f}")
    
    print("\nAverage Total Input Tokens:")
    print(f"  Method 1: {df['m1_total_input_token'].mean():.2f}")
    print(f"  Method 2: {df['m2_total_input_token'].mean():.2f}")
    print(f"  Method 3: {df['m3_total_input_token'].mean():.2f}")
    
    print("\nAverage Total Output Tokens:")
    print(f"  Method 1: {df['m1_total_output_token'].mean():.2f}")
    print(f"  Method 2: {df['m2_total_output_token'].mean():.2f}")
    print(f"  Method 3: {df['m3_total_output_token'].mean():.2f}")
    
    print("\nAverage K:")
    print(f"  Method 1: {df['m1_k'].mean():.2f}")
    print(f"  Method 2: {df['m2_k'].mean():.2f}")
    print(f"  Method 3: {df['m3_k'].mean():.2f}")
    
    print("\nAverage Judge Score:")
    print(f"  Method 1: {df['m1_judge_score'].mean():.3f}")
    print(f"  Method 2: {df['m2_judge_score'].mean():.3f}")
    print(f"  Method 3: {df['m3_judge_score'].mean():.3f}")
    
    print("\n" + "=" * 60)


def create_visualizations(df: pd.DataFrame, output_dir: str = "phase3/plots"):
    """
    Tạo các biểu đồ visualization.
    
    Args:
        df: DataFrame
        output_dir: Thư mục lưu plots
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not installed. Skipping visualizations.")
        print("Install with: pip install matplotlib")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    methods = ["Method 1 (Baseline)", "Method 2 (Adaptive Mini)", "Method 3 (Adaptive N5)"]
    
    metrics = {
        "Latency (ms)": ["m1_latency", "m2_latency", "m3_latency"],
        "Total Input Tokens": ["m1_total_input_token", "m2_total_input_token", "m3_total_input_token"],
        "Total Output Tokens": ["m1_total_output_token", "m2_total_output_token", "m3_total_output_token"],
        "K": ["m1_k", "m2_k", "m3_k"],
        "Judge Score": ["m1_judge_score", "m2_judge_score", "m3_judge_score"]
    }
    
    for metric_name, cols in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [df[cols[0]].values, df[cols[1]].values, df[cols[2]].values]
        
        bp = ax.boxplot(data, labels=methods, patch_artist=True)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel(metric_name)
        ax.set_title(f'Comparison: {metric_name}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_path / f"{metric_name.lower().replace(' ', '_')}_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")
    
    print(f"\nVisualizations saved to: {output_dir}")
    print("=" * 60)


def main():
    """
    Main function.
    """
    print("=" * 60)
    print("Phase 3: Statistics & Visualization")
    print("=" * 60)
    
    print(f"\nLoading CSV from: {config.RESULT_CSV_PATH}")
    df = load_csv()
    
    if df is None:
        return
    
    print(f"Loaded {len(df)} samples")
    
    print_statistics(df)
    
    create_visualizations(df)


if __name__ == "__main__":
    main()

