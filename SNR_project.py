############################## NECESSARY IMPORTS  ###############################
##############################################################################
##############################################################################

import numpy as np
import os
import time
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

############################## PREDICTION TEST  ###############################
##############################################################################
##############################################################################

model = VGG16(include_top=True, weights='imagenet')     ### import VGG16 model with all the weights and layers (include_top means including also last fully-connected classifaying layers)


############################## LOAD CUSTOM DATA  ###############################
##############################################################################
##############################################################################

PATH = os.getcwd() ### path to my GoogleDrive repoh
data_path = 'kaggle_bee_vs_wasp/'
data_dir_list = os.listdir(data_path)
data_dir_list.sort()
print(data_dir_list)

index = 0
data_dir_list = data_dir_list[:index] + data_dir_list[index+1 :]
print(data_dir_list)




img_data_list=[]



### LOADS IMAGES FROM ALL DIRECTORIES PRESENT IN /data
for dataset in data_dir_list:  										### loop over dir's
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:        									### loop over images inside dir
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))  	### target size for image
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		x = x/255
		print('Input image shape:', x.shape)
		img_data_list.append(x) 									### append image to list



############################## PUTTING DATA INTO RIGHT FORMAT  ###############
##############################################################################
##############################################################################
### (nr. of images, pixelsX, pixelsY, channels(RGB))

len(img_data_list)
img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
#print (img_data.shape)
img_data=np.rollaxis(img_data,1,0) 									### switch axis 0 and 1 for desired vector form
#print (img_data.shape)
img_data=img_data[0] ### get rid of '1' for desired vector form
print (img_data.shape)


############################## CLASS NR. DEF  ################################
############################## SPLITTING CLASSES  ################################
############################## SPLITTING THE DATASET ################################
##############################################################################
##############################################################################

num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

### ASSIGNING LABELS BASED ON IMG INDEX
labels[0:3183]=0
labels[3183:8126]=1
labels[8126:10564]=2
labels[10564:]=3



### DEFINING LABELS NAMES
names = ['bees','wasps','other_insects','non_insects']


Y = np_utils.to_categorical(labels, num_classes) ### convert class labels to on-hot encoding
                                                 ### -> (1,0,0,0) for class 1
                                                 ### -> (0,1,0,0) for class 2
                                                 ### -> (0,0,1,0) for class 3
                                                 ### -> (0,0,0,1) for class 4




x,y = shuffle(img_data,Y, random_state=2) ### shuffle the dataset



subX = x[:len(x)/2]
subY = y[:len(y)/2]


X_train, X_test, y_train, y_test = train_test_split(subX, subY, test_size=0.2, random_state=2) ### split the dataset



######## LOADING A MODEL AND MAKE IT COMPATIBLE WITH OUR DATASET  #############
##############################################################################
##############################################################################


image_input = Input(shape=(224, 224, 3)) ### pre-define image imput dimension

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet') ### import VGG16 model
model.summary()
last_layer = model.get_layer('fc2').output ### get the last layer (last before prediction layer)
#x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output')(last_layer) ### definition of the last prediction layer and setting it after 'fc2'
custom_vgg_model = Model(image_input, out) ### creating a model based on input and layers definiton
custom_vgg_model.summary()



################## COMPILING AND TRAINING CUSTOM MODEL FOR CLASSIFICATION ONLY  ######################
##############################################################################
##############################################################################

custom_vgg_model.compile(loss='categorical_crossentropy', optimizer='adadelta' , metrics=['accuracy'])  ### compile custom model

for layer in custom_vgg_model.layers[:-1]:  ### set all layers except last one to untrainable
	layer.trainable = False

t = time.time()
#	t = now()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=1,
							validation_data=(X_test, y_test))  ### train
print('Training time: %s' % (t - time.time()))

(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


################## COMPILING AND TRAINING CUSTOM MODEL FOR FINE TUNING  ######################
##############################################################################
##############################################################################


image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()

last_layer = model.get_layer('block5_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)
custom_vgg_model2.summary()

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False

custom_vgg_model2.summary()

custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

t=time.time()
#	t = now()
hist = custom_vgg_model2.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#custom_vgg_model2.save("./savedModels/5ep_0.5fullSET_89pc")

#### LOAD BLOCK #####
import tensorflow as tf
import keras as k
#custom_vgg_model2 = tf.keras.models.load_model("./savedModels/5ep_0.5fullSET_89pc")
#keras.models.load_model("./savedModels/5ep_0.5fullSET_89pc")
custom_vgg_model2 = k.models.load_model("./savedModels/5ep_0.5fullSET_89pc")
custom_vgg_model2.summary()
###############################

#### TAKE OUR MODEL WITHOUT CLASSIFICATION LAYER ### (UP TO FC2)
model = Model(input=custom_vgg_model2.input, output=custom_vgg_model2.get_layer('fc2').output)
model.summary()

###### EXTRACT FEATURES #######
X_train = preprocess_input(X_train)
X_valid = preprocess_input(X_test)
X_trainf = model.predict(X_train)
X_validf = model.predict(X_test)



y_train = np.argmax(y_train, axis=1) ### (to have [nr. of labes] instead of [nr.of labels , labels nr]) --- format correction
y_test = np.argmax(y_test, axis=1) ### (to have [nr. of labes] instead of [nr.of labels , labels nr]) --- format correction
print("Feature vector dimensions: ", X_trainf.shape)


### SVM CLASSIFICATION

from sklearn import svm
model1=svm.SVC(kernel='rbf').fit(X_trainf, y_train)#%--0.4249 #### feed into SVM
from keras.applications.imagenet_utils import decode_predictions

print("x",X_trainf.shape)
print("y",y_train.shape)

fit1 = model1.fit(X_trainf, y_train)

predictions_train = fit1.predict(X_trainf)

pred_train = np.round(predictions_train)


pred_acc_train = np.equal(pred_train,y_train)




pred_acc1_train = (float(np.sum(pred_acc_train)) / float(np.size(y_train)))
print("[SVM linear classifier] Accuracy on train set: ", pred_acc1_train)







predictions_test = fit1.predict(X_validf)  #### predict on features of test set

pred_test = np.round(predictions_test)
  #print("pred: ", pred_test.shape)
  #print("lab: ", y_test.shape)


pred_acc_test = np.equal(pred_test,y_test)

#print("acc: ", pred_acc_test)

pred_acc1_test = (float(np.sum(pred_acc_test)) / float(np.size(y_test)))

print("[SVM linear classifier] Accuracy on test set: ", pred_acc1_test)


