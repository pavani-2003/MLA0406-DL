from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
 
X,y=load_breast_cancer(return_X_y=True)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=23)
 
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
 
y_pred=tree.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
 
accuracy=accuracy_score(y_test,y_pred)
print('Accuracy : ',accuracy)
 
precision=precision_score(y_test,y_pred)
print("Precision : ",precision)
 
recall=recall_score(y_test,y_pred)
print("Recall : ",recall)
 
F1_score=f1_score(y_test,y_pred)
print("F1-score",F1_score)
 
 
sns.heatmap(cm,annot=True,cmap='pink',xticklabels=['malignant','benign'],yticklabels=['malignant','benign'])
plt.ylabel('Prediction')
plt.xlabel("Actual")
plt.title('Confusion Matrix')
plt.show()
