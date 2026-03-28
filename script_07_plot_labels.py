import pandas as pd
import matplotlib.pyplot as plt
 
df = pd.read_csv("Labels.csv")
 
label_counts = df["Label"].value_counts()
label_counts.plot(kind="bar")
plt.title("Glaucoma Label Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Images")
plt.show()
