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

#now it's time to try diffrent parameters
#to see which parameters are best
#for our random forest model

criterions = ["gini", "entropy", "log_loss"]
n_estimators = [75, 100, 125]
max_depths = [None, 10, 20]

best_criterion = "gini"
best_n_estimators = 75
best_max_depth = None
best_score = 0

for current_criterion in criterions:
    for current_n_estimator in n_estimators:
        for current_max_depth in max_depths:
            my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                          ('model', RandomForestClassifier(criterion = current_criterion,
                                                                           n_estimators = current_n_estimator,
                                                                           max_depth = current_max_depth,
                                                                           n_jobs = -1))
                                         ])
            
            scores = cross_validate(my_pipeline, X_train, np.array(y_train["Cover_Type"]),
                                          cv=5,
                                          scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                                          return_train_score=True, return_estimator=True)
            accurancy_scores = scores['test_accuracy']
            precision_scores = scores['test_precision_macro']
            recall_scores = scores['test_recall_macro']
            f1_scores = scores['test_f1_macro']

            print("criterion =", current_criterion,
                  "n_estimators =", current_n_estimator,
                  "max_depth =", current_max_depth)
            print("accurancy_scores:\n", accurancy_scores)
            print("precision_scores:\n", precision_scores)
            print("recall_scores:\n", recall_scores)
            print("f1_scores:\n", f1_scores)
            
            score = np.mean(accurancy_scores)
            print("Average accurancy score:", score)
            if(score > best_score):
                best_criterion = current_criterion
                best_n_estimators = current_n_estimator
                best_max_depth = current_max_depth
                best_score = score

print("best_score =", best_score)
print("best_criterion =", best_criterion)
print("best_n_estimators =", best_n_estimators)
print("best_max_depth =", best_max_depth)

#now that we've got our best parameters
#it's time to test it on test data and see
#how it works
rf = RandomForestClassifier(criterion = best_criterion,
                            n_estimators = best_n_estimators,
                            max_depth = best_max_depth,
                            n_jobs = -1)
rf.fit(X_train, np.array(y_train["Cover_Type"]))

y_pred = rf.predict(X_train)
confusion_mat = confusion_matrix(np.array(y_train["Cover_Type"]), y_pred)
print(confusion_mat)

#time for testing
y_pred = rf.predict(X_test)
confusion_mat = confusion_matrix(np.array(y_test["Cover_Type"]), y_pred)
print(confusion_mat)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#now let's try it again for decision tree model
#finding best parameters
criterions = ["gini", "entropy", "log_loss"]
splitters = ["best", "random"]
max_depths = [None, 10, 20]

best_criterion = "gini"
best_splitter = "best"
best_max_depth = None
best_score = 0

for current_criterion in criterions:
    for current_splitter in splitters:
        for current_max_depth in max_depths:
            my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                          ('model', DecisionTreeClassifier(criterion = current_criterion,
                                                                           splitter = current_splitter,
                                                                           max_depth = current_max_depth))
                                         ])
            
            scores = cross_validate(my_pipeline, X_train, np.array(y_train["Cover_Type"]),
                                          cv=5,
                                          scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                                          return_train_score=True, return_estimator=True)
            accurancy_scores = scores['test_accuracy']
            precision_scores = scores['test_precision_macro']
            recall_scores = scores['test_recall_macro']
            f1_scores = scores['test_f1_macro']

            print("criterion =", current_criterion,
                  "splitter =", current_splitter,
                  "max_depth =", current_max_depth)
            print("accurancy_scores:\n", accurancy_scores)
            print("precision_scores:\n", precision_scores)
            print("recall_scores:\n", recall_scores)
            print("f1_scores:\n", f1_scores)
            
            score = np.mean(accurancy_scores)
            print("Average accurancy score:", score)
            if(score > best_score):
                best_criterion = current_criterion
                best_splitter = current_splitter
                best_max_depth = current_max_depth
                best_score = score

print("best_score =", best_score)
print("best_criterion =", best_criterion)
print("best_splitter =", best_splitter)
print("best_max_depth =", best_max_depth)

#testing best model on test data
dt = DecisionTreeClassifier(criterion = best_criterion,
                            splitter = best_splitter,
                            max_depth = best_max_depth)
dt.fit(X_train, np.array(y_train["Cover_Type"]))

y_pred = dt.predict(X_train)
confusion_mat = confusion_matrix(np.array(y_train["Cover_Type"]), y_pred)
print(confusion_mat)

y_pred = dt.predict(X_test)
confusion_mat = confusion_matrix(np.array(y_test["Cover_Type"]), y_pred)
print(confusion_mat)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)