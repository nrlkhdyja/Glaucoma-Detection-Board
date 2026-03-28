import pandas as pd
 
df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])
 
filtered_df = df[df["Quality Score"] >= 5]
 
print("Original dataset size:", len(df))
print("Filtered dataset size:", len(filtered_df))
