import numpy as np
import pandas as pd
 
dataset = pd.read_csv("C:\\Users\\pavani.k\\OneDrive\\Desktop\\pavani college\\pavani project\\Dataset\\d1.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 2)
 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
 
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy=accuracy_score(y_test, y_pred)
print(accuracy)
