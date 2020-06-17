from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
print(x_train.shape, '\n')
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=x_train[0].shape))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(units=757, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
print(model.summary())
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
trained_model = model.fit(x_train, y_train, verbose=1, epochs=3)
print(trained_model.history.keys())
plt.style.use("seaborn")
fig, ax = plt.subplots(figsize=(15, 9))
ax.plot(trained_model.history["loss"])
ax.plot(trained_model.history["accuracy"])
ax.legend(["Loss", "Accuracy"])
ax.set(title="LOSS VS ACCURACY GRAPH", xlabel="EPOCHS", ylabel="LOSS & ACCURACY")
plt.show()
y_pred = model.predict_classes(x_test)
cf = classification_report(y_test, y_pred)
print(cf)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15, 9))
sns.heatmap(cm, annot=True)
plt.show()
acs = model.evaluate(x_test, y_test)[1]
acs = acs * 100
print(acs)
plt.imshow(x_train[5].reshape(x_train[5].shape[0], x_train[5].shape[1]))
plt.title("Actual = {} Prediction = {}".format(y_test[5], y_pred[5]))
plt.show()