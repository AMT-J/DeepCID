import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (1, 5), activation='relu', padding='SAME')
        self.pool1 = tf.keras.layers.MaxPooling2D((1, 2), strides=(1, 2), padding='SAME')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        
        self.conv2 = tf.keras.layers.Conv2D(64, (1, 5), activation='relu', padding='SAME')
        self.pool2 = tf.keras.layers.MaxPooling2D((1, 2), strides=(1, 2), padding='SAME')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(2)

    def call(self, x, training=False):
        x = tf.reshape(x, [-1, 1, 881, 1])
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout3(x, training=training)
        x = self.fc2(x)
        return tf.nn.softmax(x, axis=-1)

# Instantiate the model
model = MyModel()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Load and preprocess data
path = '.'
compound = 0

datafile1 = os.path.join(path, 'augmented_data', f'{compound}component.npy')
X = np.load(datafile1)
Xtrain0 = X[0:15000]
Xvalid0 = X[15000:17500]
Xtest0 = X[17500:20000]
scaler = preprocessing.StandardScaler().fit(Xtrain0)
Xtrain = scaler.transform(Xtrain0)
Xvalid = scaler.transform(Xvalid0)
Xtest = scaler.transform(Xtest0)

datafile2 = os.path.join(path, 'augmented_data', f'{compound}label.npy')
Y1 = np.load(datafile2)
Y2 = np.ones_like(Y1) - Y1
Y = np.concatenate((Y1, Y2), axis=1)
Ytrain = Y[0:15000]
Yvalid = Y[15000:17500]
Ytest = Y[17500:20000]

# Train the model
batch_size = 100
epochs = 300

history = model.fit(Xtrain, Ytrain, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_data=(Xvalid, Yvalid),
                    verbose=1)

# Save the model
save_path = './saved_model'
model.save(save_path)

# Plot accuracy and loss
TIMES = [(i + 1) for i in range(epochs)]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(TIMES, history.history['accuracy'], 'r', label='Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax2 = ax1.twinx()
ax2.plot(TIMES, history.history['loss'], 'g', label='Loss')
ax2.set_ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(Xtest, Ytest, verbose=2)
print('The test accuracy %.1f%%' % (test_acc * 100))
