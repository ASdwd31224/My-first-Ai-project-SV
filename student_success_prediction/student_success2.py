X = [
    [1, 45],
    [2, 50],
    [2, 55],
    [3, 60],
    [3, 65],
    [4, 70],
    [4, 75],
    [5, 72],
    [5, 80],
    [6, 85],
    [6, 90],
    [7, 88],
    [7, 95],
    [8, 92],
    [9, 98]
]

y = [
    0, 0, 0, 0, 0,
    1, 1, 0, 1, 1,
    1, 1, 1, 1, 1
]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, recall_score, precision_score

acc = accuracy_score(y_test, predict)
rec = recall_score(y_test, predict)
pre = precision_score(y_test, predict)

from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier()
model_tree.fit(x_train, y_train)
predict_tree = model_tree.predict(x_test)

acc_tree = accuracy_score(y_test, predict_tree)
rec_tree = recall_score(y_test, predict_tree)
pre_tree = precision_score(y_test, predict_tree)

from sklearn.naive_bayes import GaussianNB

model_naive = GaussianNB()
model_naive.fit(x_train, y_train)
predict_naive = model_naive.predict(x_test)

acc_naive = accuracy_score(y_test, predict_naive)
rec_naive = recall_score(y_test, predict_naive)
pre_naive = precision_score(y_test, predict_naive)

print("\n ---===New student===---")

new_st = [[4, 30]]

print("KNN:", model.predict(new_st))
print("Tree:", model_tree.predict(new_st))
print("Naive:", model_naive.predict(new_st))
