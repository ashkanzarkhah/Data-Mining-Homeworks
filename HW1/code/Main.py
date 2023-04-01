import pandas as pd
from sklearn.utils import shuffle

#Reading dataset
df = pd.read_csv("../Dataset/covtype.csv")

#Shuffeling data set so that picking last 30% of objects been equal to randomly picking 30% of dataset
df = shuffle(df)

#Deviding dataset into X and y
columns = df.columns
X = df.filter(items = columns[0:-1])
y = df.filter(items = columns[-1:])

#Picking X_train and X_test and y_train and y_test
train_rows_count = int(df.shape[0] * 0.7)
test_rows_count = df.shape[0] - train_rows_count
X_train = X.head(train_rows_count)
X_test = X.tail(test_rows_count)
y_train = y.head(train_rows_count)
y_test = y.tail(test_rows_count)
#print(X.head)
