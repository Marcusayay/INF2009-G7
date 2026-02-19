import pandas as pd 


try1 = {
    "model1": {"accuracy": 0.95, "precision": 0.92, "recall": 0.93},
    "model2": {"accuracy": 0.96, "precision": 0.91, "recall": 0.94}, 
}

df = pd.DataFrame(try1).T
print(df) 