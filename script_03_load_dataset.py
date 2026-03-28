import os

print("Current directory:", os.getcwd())
print("Files here:", os.listdir())

import pandas as pd
 
# Load dataset
df = pd.read_csv("Labels.csv")
 
print("Dataset successfully loaded.\n")
print("First 5 rows of dataset:\n")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nColumn names:")
print(df.columns)

