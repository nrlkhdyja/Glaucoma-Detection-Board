import pandas as pd
from sklearn.model_selection import train_test_split
 
df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])
 
df["label_numeric"] = df["Label"].map({"GON+": 1, "GON-": 0})
 
# Get unique patient IDs
patients = df["Patient"].unique()
 
# Split patients into train/test
train_patients, test_patients = train_test_split(
    patients,
    test_size=0.2,
    random_state=42
)
 
# Select images belonging to those patients
train_df = df[df["Patient"].isin(train_patients)]
test_df = df[df["Patient"].isin(test_patients)]
 
print("Training images:", len(train_df))
print("Testing images:", len(test_df))
print("Training patients:", train_df["Patient"].nunique())
print("Testing patients:", test_df["Patient"].nunique())

train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)
print("Datasets saved successfully.")
