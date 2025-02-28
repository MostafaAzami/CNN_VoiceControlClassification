import matplotlib.image as mpimg
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import random
import matplotlib.pyplot as plt
 

def Create_DB():
    img_dim = 80
    DBArray = np.zeros([1,img_dim,img_dim], dtype=int)
    lable = np.zeros([1], dtype=int)
    filenamelable = np.zeros([1], dtype=int)
    DBArray= np.delete(DBArray,0,0)
    lable= np.delete(lable,0)
    filenamelable= np.delete(filenamelable,0)

    DATA_PATHS = {
        "B": 0,
        "F": 1,
        "L": 2,
        "R": 3,
        "S": 4
    }
    DB_PATH = "E:/2-voice/implementation/DB/"
    for category, label in DATA_PATHS.items():
        folder = os.path.join(DB_PATH, category, "grayimg-ag80-v2")
        for file in os.listdir(folder):
            filepath = os.path.join(folder, file)
            if os.path.isfile(filepath):       
                img=mpimg.imread(filepath)
                img = img.reshape(1,img_dim,img_dim)
                DBArray = np.append(DBArray,img, axis=0)   
                lable = np.append(lable,[lable], axis=0)   
                filenamelable = np.append(filenamelable,[file], axis=0) 

def DataArgumentation(train_dataset):
    ###################### data_augmentation ##############
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    
    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
          ax = plt.subplot(3, 3, i + 1)
          augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
          plt.imshow(augmented_image[0] / 255)
          plt.axis('off')
     
################################ read data
(DBArray1 , lable , filenamelable) = Create_DB()
sizesample = lable.shape[0]
randoindex = range(0,sizesample)
randoindex = random.sample(randoindex, k=sizesample)
DBArray = DBArray1[randoindex]
lable = lable[randoindex]
filenamelable = filenamelable[randoindex]
sizetest = 70
X_test = DBArray[0:sizetest]
y_test = lable[0:sizetest]
filenamelable_test = filenamelable[0:sizetest]
filenamelable_train = filenamelable[sizetest:sizesample]
y_train = lable[sizetest:sizesample]
X_train = DBArray[sizetest:sizesample]
X_trainindex = range(0,sizesample-sizetest)
c= 0
for name in filenamelable_test:
    isarg = name.find("ag")
    if(isarg>=0):
        name = name[4:]
    matching =  np.flatnonzero(np.core.defchararray.find(filenamelable_train,name)!=-1)
    for i in range (0,len(matching)):
        filenamelable_train = np.delete(filenamelable_train, matching[i]-i)
        X_trainindex = np.delete(X_trainindex, matching[i]-i)
        y_train = np.delete(y_train, matching[i]-i)
        c= c+1
        
X_train = X_train[X_trainindex]
print('Num Data:' , DBArray.shape)
print("size train:", X_train.shape)
print("size test:", X_test.shape)
################################ end read data

X_train = X_train.reshape(y_train.size,80,80,1)
X_test = X_test.reshape(y_test.size ,80,80,1)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#Building the model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(80,80,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add( Dropout(.25))
model.add(Dense(5, activation='softmax'))
#Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Training the model
history =model.fit(X_train, y_train,validation_data=(X_test, y_test), batch_size=10, epochs=5)
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))


################################# accuracy plot##############
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend() 
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

