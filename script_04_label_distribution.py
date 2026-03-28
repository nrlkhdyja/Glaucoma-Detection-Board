import os

print("Current directory:", os.getcwd())
print("Files here:", os.listdir())

import pandas as pd
 
df = pd.read_csv("Labels.csv")
 
print("Label distribution:\n")
print(df["Label"].value_counts())
