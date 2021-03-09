import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# reading the data using pandas
data = pd.read_csv("car.data")

# changing data on CSV yo numerical value using sklearn
le = preprocessing.LabelEncoder()
buying =le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform((list(data["safety"])))
cls = le.fit_transform((list(data["class"])))
persons = le.fit_transform(list(data["persons"]))
door =le.fit_transform(list(data["door"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot,safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predict = model.predict(x_test)
names = ["unacc", "acc", "good", "verygood"]

for x in range(len(predict)):
    print("predict:", names[predict[x]], "Data:", x_test[x], names[y_test[x]])
    n = model.kneighbors([x_test[x]], 7, True)
    print("N:", n)