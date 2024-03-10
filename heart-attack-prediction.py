import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# I took the dataset from Kaggle
Path = "dataset.csv"
heart = pd.read_csv(Path)
print(heart)
heart.info()

data = pd.DataFrame(heart)
X = data.drop('target', axis=1)
Y = data['target']
print(X)
print(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

accuracies = {}
models = {'SVM' : svm.SVC(),
         'KNN' : KNeighborsClassifier(),
         'Random Frorest' : RandomForestClassifier(),
         'Decision Tree' : tree.DecisionTreeClassifier(),
         'Logistic Regression' : LogisticRegression()
         }
for name, model in models.items():
    print(f"Training {name}")
    model.fit(X,Y)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy}")
    print(classification_report(y_test, y_pred))
    confusion_matrix(y_test, y_pred)
    accuracies[name] = accuracy

plt.figure(figsize=(10,6))
bars = plt.barh(list(accuracies.keys()), list(accuracies.values()), color='pink')

for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.2f}', 
             va='center', ha='left', fontsize=10, color='black')
plt.xlabel('Accuracy')
plt.title('Accuracies of different models')
plt.savefig('Models-Accuracies.png')
plt.show()




