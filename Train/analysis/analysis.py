import pandas as pd 
from tabulate import tabulate 

try:
    df = pd.read_csv("Train/analysis/benchmark_results_finetuned_1.csv")
except: 
    df = pd.read_csv("benchmark_results_finetuned_1.csv")

def display_pretty_table(df, title):
    """Helper to print a formatted table with a header in BOLD and COLOR."""
    # ANSI escape codes
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    
    print(f"\n{CYAN}{BOLD}{'='*20} {title} {'='*20}{RESET}")
    
    # Generate the table string
    table_str = tabulate(df, headers='keys', tablefmt='psql', showindex=False)
    
    # Print the table in BOLD so it appears thicker/more visible
    print(f"{BOLD}{table_str}{RESET}")
    
    print(f"{CYAN}{BOLD}{'=' * (42 + len(title))}{RESET}\n")

df.sort_values(by="system_fps_mean", ascending=False, inplace=True) 
# remove fam coluumn 
df.drop(columns=['family'], inplace=True)
display_pretty_table(df, "ALL BENCHMARK RESULTS")