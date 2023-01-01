from si.src.si.io.CSV import read_csv
from si.src.si.linear_model.logistic_regression import LogisticRegression
from sklearn.preprocessing import StandardScaler
from si.src.si.model_selection.split import train_test_split
from si.src.si.feature_extraction.k_mer_new import KMerNew

dataset = read_csv(r'C:\Users\HP-PC\PycharmProjects\pythonProject2\si\datasets\transporters.csv', features = False, label = True)
alfabeto = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
k_mer_new = KMerNew(k = 2, alfabeto = alfabeto)

dataset_ = k_mer_new.fit_transform(dataset)
print(dataset_.X)
print(dataset_.features)

dataset_.X = StandardScaler().fit_transform(dataset_.X)

dataset_train, dataset_test = train_test_split(dataset_, test_size=0.3)

logistic = LogisticRegression()
logistic.fit(dataset_train)

score = logistic.score(dataset_test)
print({score})

