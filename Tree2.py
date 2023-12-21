import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

feature = ['toan', 'ly', 'anh', 'ket qua']
properties = [[10, 8, 5, 'do'],
              [7, 8, 5, 'truot'],
              [9, 6, 5, 'truot'],
              [9.5, 7, 9, 'do'],
              [6, 4, 10, 'truot'],
              [1, 10, 10, 'do'],
              [6, 9.5, 6.5, 'do'],
              [9, 8, 4, 'do'],
              [10, 3, 5, 'truot'],
              [7, 8, 7, 'do'],
              [5, 9, 6, 'truot'],
              [6, 1, 6.5, 'truot'],
              [7.5, 9.5, 6, 'do'],
              [10, 9, 9, 'do'],
              [6, 9, 10, 'truot']]

df = pd.DataFrame(data = properties, columns = feature)
print(df)

X_train = df.drop('ket qua', axis = 1)
Y_train = df['ket qua']

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

tree_rules = export_text(model, feature_names = list(X_train.columns))
print('Decision Tree Rules: \n', tree_rules)

print('Nhap diem cac mon')
a = float(input('--diem toan: '))
b = float(input('--diem ly: '))
c = float(input('--diem tieng anh: '))
X_test = [[a, b, c]]

prediction = model.predict(X_test)
print('Ket qua cua hoc sinh nay la: ', prediction[0])