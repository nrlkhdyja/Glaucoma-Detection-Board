import pandas as pd
 
df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])
 
df["label_numeric"] = df["Label"].map({"GON+": 1, "GON-": 0})
df.to_csv("glaucoma_clean_dataset.csv", index=False)
print("Clean dataset saved.")
