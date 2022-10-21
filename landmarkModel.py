#https://www.sharpsightlabs.com/blog/scikit-train_test_split/
#https://github.com/kinivi/hand-gesture-recognition-mediapipe/blob/main/keypoint_classification.ipynb

import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = 'vid1/output.csv'
model_save_path = 'vid1/landmarkClassifier.hdf5'

NUM_CLASSES = 3 #number of gestures
RANDOM_SEED = 42

features = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1))) #update for my csv layout
labels = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
featuresTrain, featuresTest, labelsTrain, labelsText = train_test_split(features, features, train_size=0.75, random_state=RANDOM_SEED)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)

es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    featuresTrain,
    labelsTrain,
    epochs=1000,
    batch_size=128,
    validation_data=(featuresTest, labelsText),
    callbacks=[cp_callback, es_callback]
)

val_loss, val_acc = model.evaluate(featuresTest, labelsText, batch_size=128)
model = tf.keras.models.load_model(model_save_path)

predict_result = model.predict(np.array([featuresTest[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

model.save(model_save_path, include_optimizer=False)