#https://www.sharpsightlabs.com/blog/scikit-train_test_split/
#https://github.com/kinivi/hand-gesture-recognition-mediapipe/blob/main/keypoint_classification.ipynb

import csv
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools

classMap = {
'tap_on_distal_phalanx_of_index_finger_using_thumb': '0',
'one': '1', 
'double_tap_on_distal_phalanx_for_ring_finger_with_thumb': '10',
'index_finger_swipe_right': '11',
'index_finger_swipe_down': '12',
'tap_on_distal_phalanx_of_last_finger_using_thumb': '13',
'tap_on_distal_phalanx_of_middle_finger_using_thumb': '14',
'index_finger_single_tap': '15', 
'hand_close': '16', 
'tap_on_middle_phalanx_of_index_finger_using_thumb': '17', 
'two': '18', 
'tap_on_proximal_phalanx_of_index_finger_using_thumb': '19',
'double_tap_on_middle_phalanx_for_index_finger_with_thumb': '2',
'zoom_out_with_index_finger_and_thumb': '20', 
'double_tap_on_middle_phalanx_for_last_finger_using_thumb': '21',
'index_finger_swipe_up': '22',
'rub_thumb_on_index_finger_forward': '23',
'rub_thumb_on_index_finger_backward': '24', 
'tap_on_distal_phalanx_of_ring_finger_with_thumb': '25', 
'double_tap_on_distal_phalanx_for_last_finger_using_thumb': '26', 
'three': '27', 
'double_tap_on_proximal_phalanx_of_last_finger_using_thumb': '28', 
'index_finger_double_tap': '29',
'select_with_thumb_and_finger': '3', 
'rotate_index_finger_anti-clockwise': '30', 
'double_tap_on_distal_phalanx_for_index_finger_using_thumb': '31',
'rub_on_index_finger_with_thumb_clockwise': '32', 
'double_tap_on_proximal_phalanx_for_index_finger_using_thumb': '33',
'tap_on_middle_phalanx_of_middle_finger_using_thumb': '34',
'tap_on_proximal_phalanx_of_last_finger_using_thumb': '35',
'tap_on_middle_phalanx_of_last_finger_using_thumb': '36',
'index_finger_swipe_left': '37',
'tap_on_proximal_phalanx_of_ring_finger_using_thumb': '38',
'four': '39',
'rub_on_index_finger_with_thumb_anti-clockwise': '4',
'double_tap_on_proximal_phalanx_for_ring_finger_with_thumb': '40',
'zoom_out_with_fingers': '41',
'rotate_index_finger_clockwise': '42',
'double_tap_on_distal_phalanx_for_middle_finger_using_thumb': '43',
'double_tap_on_middle_phalanx_for_middle_finger_using_thumb': '44',
'zoom_in_with_fingers': '45',
'five': '46',
'double_tap_on_middle_phalanx_for_ring_finger_with_thumb': '47',
'snap': '48',
'double_tap_on_proximal_phalanx_for_middle_finger_using_thumb': '5',
'tap_on_middle_phalanx_of_ring_finger_with_thumb': '6',
'tap_on_proximal_phalanx_of_middle_finger_using_thumb': '7', 
'hand_open': '8',
'zoom_in_with_index_finger_and_thumb': '9'
}

dataset = '/home/exx/hannah/GitProjects/microgesture/processedImages.csv'
model_save_path = '/home/exx/hannah/GitProjects/microgesture/landmarkClassifier.hdf5'

NUM_CLASSES = 49 #number of gestures
RANDOM_SEED = 42
LANDMARKS = 2100 #21 * 2 * frames gathered

def PlotCM(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
		'''
			This function code is from: https://deeplizard.com/learn/video/0LhiS6yu2qQ
		'''
		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')
	
		print(cm)
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

# featuresArray = np.loadtxt(fname=dataset, delimiter=':', dtype='str', usecols=(2), skiprows=1) #update for my csv layout
labels = np.loadtxt(fname=dataset, delimiter=',', dtype='int32', usecols=(0), skiprows=1)
features = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, LANDMARKS + 1)), skiprows=1)

featuresTrain, featuresTest, labelsTrain, labelsText = train_test_split(features, labels, train_size=0.75, random_state=RANDOM_SEED)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(LANDMARKS),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)

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
    callbacks=[cp_callback]
)

val_loss, val_acc = model.evaluate(featuresTest, labelsText, batch_size=128)
model = tf.keras.models.load_model(model_save_path)

predict_result = model.predict(np.array([featuresTest[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

Y_pred = model.predict(featuresTest)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(labelsText, y_pred)
# print_confusion_matrix(labelsText, y_pred)
PlotCM(cm, classMap.values(), normalize=False, title='Accuracy')
plt.savefig('/home/exx/hannah/GitProjects/microgesture/testCM.png')

model.save(model_save_path, include_optimizer=False)