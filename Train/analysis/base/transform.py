# add this shit in 
# # f_family['size'] = df_family["model"].apply(lambda x: x.split("_")[0][-1] if "_" in x else x.split(".")[0][-1])
 
import pandas as pd 

df = pd.read_csv("benchmark_results.csv")  
df['size'] = df["model"].apply(lambda x: x.split("_")[0][-1] if "_" in x else x.split(".")[0][-1]) 
df.to_csv("benchmark_results_with_size.csv", index=False)