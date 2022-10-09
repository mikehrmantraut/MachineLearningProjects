import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# read data
df = pd.read_csv("Iris.csv")

# review data
df.head()
df.info
df.shape

# renaming species' names
df["Species"] = np.where(df["Species"] == 'Iris-setosa', "setosa", df["Species"])
df["Species"] = np.where(df["Species"] == 'Iris-virginica', "virginica", df["Species"])
df["Species"] = np.where(df["Species"] == 'Iris-versicolor', "versicolor", df["Species"])

df['Species'].value_counts()

sns.FacetGrid(df, hue="Species", height=6).map(plt.scatter, "PetalLengthCm", "SepalWidthCm").add_legend()
plt.show()

flower_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
df["Species"] = df["Species"].map(flower_mapping)

# preparing inputs and outputs
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df[['Species']].values


model = LogisticRegression()
model.fit(X, y.ravel())

# Accuracy
model.score(X, y)

# Make predictions
expected = y
predicted = model.predict(X)

# 0.97 accuracy
print(metrics.classification_report(expected, predicted))
# 0-> 50/50 1-> 47/50 2-> 49/50
print(metrics.confusion_matrix(expected, predicted))
