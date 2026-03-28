import pandas as pd
 
df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])
 
label_map = {
    "GON+": 1,
    "GON-": 0
}
 
df["label_numeric"] = df["Label"].map(label_map)
print(df.head())