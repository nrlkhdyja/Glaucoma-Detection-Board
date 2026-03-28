import pandas as pd
 
df = pd.read_csv("Labels.csv")
 
print("Total number of images:")
print(len(df))
print("\nTotal unique patients:")
print(df["Patient"].nunique())
print("\nImages per patient:")
print(df.groupby("Patient").size().head())
