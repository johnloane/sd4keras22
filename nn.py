import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def plot_decision_boundary(X, labels, model):
    x_span = np.linspace(min(X[:, 0])-1, max(X[:, 0])+1, 50)
    y_span = np.linspace(min(X[:, 1])-1, max(X[:, 1])+1, 50)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)



n_pts = 500
np.random.seed(0)

Xa = np.array([np.random.normal(13, 2, n_pts), np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts), np.random.normal(6, 2, n_pts)]).T

X = np.vstack((Xa, Xb))
labels = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)
#print(labels)




model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
adam = Adam(learning_rate=0.1)
model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
h = model.fit(x=X, y=labels, verbose=1, batch_size=50, epochs=20, shuffle='true')
# plt.plot(h.history['loss'])
# plt.title('loss')
# plt.xlabel('epoch')
# plt.legend(['loss'])
# plt.show()


plot_decision_boundary(X, labels, model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
x = 7.5
y = 5

point = np.array([[x, y]])
prediction = model.predict(point)

plt.plot([x], [y], marker="x", markersize=10, color="black")
print("Prediction is: ", prediction)
plt.show()







