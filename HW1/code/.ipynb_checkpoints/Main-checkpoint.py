import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Reading dataset
df = pd.read_csv("../Dataset/covtype.csv")
print(df.head)

#Shuffeling data set so that picking last 30% of objects been equal to randomly picking 30% of dataset
df = shuffle(df)
print(df.head)

#Deviding dataset into X and y
columns = df.columns
X = df.filter(items = columns[0:-1])
y = df.filter(items = columns[-1:])
print(X.head)
print(y.head)

#Picking X_train and X_test and y_train and y_test
train_rows_count = int(df.shape[0] * 0.7)
test_rows_count = df.shape[0] - train_rows_count
X_train = X.head(train_rows_count)
X_test = X.tail(test_rows_count)
y_train = y.head(train_rows_count)
y_test = y.tail(test_rows_count)
print("X_train size:", X_train.shape, ", X_test size:", X_test.shape)
print("y_train size:", y_train.shape, ", y_test size:", y_test.shape)

#Cheking data for preprocessing
#First checking number of missing values
print("Number of missing values in total:", df.isnull().sum().sum())
#Second watching how or data varies for scaling
for column in columns[:-1]:
    print(column, "Max:", X[column].max(), "Min:", X[column].min())
#now that we see some of them are binary but some of them vary
#a lot it is got to normalize all of them
#we use minmax scaling and becuase we can't use it splitly in X_train and X_test
#we do it on X and devide it again
scaler = MinMaxScaler() 
X.iloc[:,:] = scaler.fit_transform(X.iloc[:,:].to_numpy())
X_train = X.head(train_rows_count)
X_test = X.tail(test_rows_count)
#now we check how our data varies now
for column in columns[:-1]:
    print(column, "Max:", X[column].max(), "Min:", X[column].min())


print (y_train['Cover_Type'][0])
#now it's time to learn
#rf = RandomForestClassifier()
#rf.fit(X_train, y_train[:][0])

#time for testing
#y_pred = rf.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)




