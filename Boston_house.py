import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = 'boston_train.csv'
df = pd.read_csv(file_path)

df.dropna()
df = df.drop(['ID'], axis=1)


X = df.drop('medv', axis=1)
y = df['medv']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()


history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=30)


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history["mae"]
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


pred = model.predict(X_test_scaled)

pred_values = []

for i in pred:
    num = i[0]
    pred_values.append(num)


x_values = []

l = [x_values.append(i) for i in range(67)]


y_test_real_values = []

for i in y_test:
    y_test_real_values.append(i)


pred_values1 = pred_values[0: 10]
pred_values2 = pred_values[10: 20]
pred_values3 = pred_values[20: 30]
pred_values4 = pred_values[30: 40]
pred_values5 = pred_values[40: 50]
pred_values6 = pred_values[50: 67]

yt_values1 = y_test_real_values[0: 10]
yt_values2 = y_test_real_values[10: 20]
yt_values3 = y_test_real_values[20: 30]
yt_values4 = y_test_real_values[30: 40]
yt_values5 = y_test_real_values[40: 50]
yt_values6 = y_test_real_values[50: 67]


x_list1 = x_values[0: 10]
x_list2 = x_values[10: 20]
x_list3 = x_values[20: 30]
x_list4 = x_values[30: 40]
x_list5 = x_values[40: 50]
x_list6 = x_values[50: 67]


plt.title('Prediction vs Real Value 1 of 6')
plt.scatter(x_list1, pred_values1, label='Predticted data', color='r')
plt.scatter(x_list1, yt_values1, label='Real data', color='b')
plt.xlabel('X number in List')
plt.ylabel('medv')
plt.legend()
plt.show()

plt.title('Prediction vs Real Value 2 of 6')
plt.scatter(x_list2, pred_values2, label='Predticted data', color='r')
plt.scatter(x_list2, yt_values2, label='Real data', color='b')
plt.xlabel('X number in List')
plt.ylabel('medv')
plt.legend()
plt.show()

plt.title('Prediction vs Real Value 3 of 6')
plt.scatter(x_list3, pred_values3, label='Predticted data', color='r')
plt.scatter(x_list3, yt_values3, label='Real data', color='b')
plt.xlabel('X number in List')
plt.ylabel('medv')
plt.legend()
plt.show()

plt.title('Prediction vs Real Value 4 of 6')
plt.scatter(x_list4, pred_values4, label='Predticted data', color='r')
plt.scatter(x_list4, yt_values4, label='Real data', color='b')
plt.xlabel('X number in List')
plt.ylabel('medv')
plt.legend()
plt.show()

plt.title('Prediction vs Real Value 5 of 6')
plt.scatter(x_list5, pred_values5, label='Predticted data', color='r')
plt.scatter(x_list5, yt_values5, label='Real data', color='b')
plt.xlabel('X number in List')
plt.ylabel('medv')
plt.legend()
plt.show()

plt.title('Prediction vs Real Value 6 of 6')
plt.scatter(x_list6, pred_values6, label='Predticted data', color='r')
plt.scatter(x_list6, yt_values6, label='Real data', color='b')
plt.xlabel('X number in List')
plt.ylabel('medv')
plt.legend()
plt.show()

