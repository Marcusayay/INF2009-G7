import pandas as pd
from tabulate import tabulate


try:
    df = pd.read_csv("Train/analysis/benchmark_results.csv") 
except: 
    df = pd.read_csv("benchmark_results.csv") 
print(df[['family']].value_counts())


def display_pretty_table(df, title):
    """Helper to print a formatted table with a header."""
    print(f"\n{'='*20} {title} {'='*20}")
    # Using 'psql' format for a clean, professional look
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    print("=" * (42 + len(title)) + "\n")

def family_analysis():
    df_fam = df.groupby("family").agg({ 
        "system_fps_mean": "mean", 
        "system_latency_mean_ms": "mean", 
        "model_inference_ms": "mean", 
        "python_overhead_ms": "mean",
        "max_ram_percent": "mean", 
        "max_cpu_temp_c": "mean",
        "model": "count" 
    }).reset_index() 

    df_fam = df_fam.sort_values(by="system_fps_mean", ascending=False) 
    display_pretty_table(df_fam, "AVERAGE BENCHMARK RESULTS BY FAMILY")

    best_models = df.loc[df.groupby("family")["system_fps_mean"].idxmax()] 
    display_pretty_table(best_models[["model", "family", "system_fps_mean", "system_latency_mean_ms"]], "BEST MODEL IN EACH FAMILY")


def size_analysis(family): 
    df_family = df[df['family'] == family] 

    df_size = df_family.groupby("size").agg({ 
        "system_fps_mean": "mean", 
        "system_latency_mean_ms": "mean", 
        "model_inference_ms": "mean", 
        "python_overhead_ms": "mean",
        "max_ram_percent": "mean", 
        "max_cpu_temp_c": "mean",
        "model": "count" 
    }).reset_index() 

    df_size = df_size.sort_values(by="system_fps_mean", ascending=False) 
    display_pretty_table(df_size, f"AVERAGE BENCHMARK RESULTS FOR {family.upper()} BY SIZE")

    best_models = df_family.loc[df_family.groupby("size")["system_fps_mean"].idxmax()] 
    display_pretty_table(best_models[["model", "size", "system_fps_mean", "system_latency_mean_ms"]], f"BEST MODEL IN EACH SIZE FOR {family.upper()}")  

def export_type_analysis(family,size): 
    df_filtered = df[(df["family"] == family) & (df["size"] == size)]

    df_sorted = df_filtered.sort_values(by="system_fps_mean", ascending=False)
    # drop frames_measured, size and family
    df_sorted = df_sorted.drop(columns=["frames_measured", "size", "family"], errors='ignore') 
    display_pretty_table(
        df_sorted, 
        f"{family.upper()} SIZE {size.upper()} BY EXPORT TYPE"
    )


family_analysis()
size_analysis("yolo26")
export_type_analysis("yolo26", "n") 