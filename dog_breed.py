import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.preprocessing import image
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam


# 填label.csv位置
labels_all = pd.read_csv("labels.csv")

# 計算breed的數量
breeds_all = labels_all["breed"]
# breed_counts = breeds_all.value_counts()
# print(breed_counts.head(10))

# # 取得所有breed的名稱
# breed_name = labels_all["breed"].unique()

# 只取得前10多的breed
breed_name = ['scottish_deerhound','maltese_dog',
              'afghan_hound','entlebucher',"bernese_mountain_dog"
              ,"shih-tzu","great_pyrenees","pomeranian","basenji","samoyed"]

labels = labels_all[(labels_all['breed'].isin(breed_name))]
labels = labels.reset_index()


X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')
Y_data = label_binarize(labels['breed'], classes = breed_name)


for i in tqdm(range(len(labels))):
    img = image.load_img(f'dog-breed-identification//train//{labels["id"][i]}.jpg',
                          target_size=(224, 224))
    img = image.img_to_array(img)
    x = np.expand_dims(img.copy(), axis=0)
    X_data[i] = x / 255.0
    

# Printing train image and one hot encode shape & size
print('\nTrain Images shape: ',X_data.shape,' size: {:,}'.format(X_data.size))
print('One-hot encoded output shape: ',Y_data.shape,' size: {:,}'.format(Y_data.size))



# Building the Model


model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=16,
                 kernel_size=(3,3),padding="same",
                 activation="relu"))
model.add(Conv2D(filters=16,kernel_size=(3,3),
                 padding="same",
                 activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), 
                 padding="same",
                  activation="relu"))
model.add(Conv2D(filters=32, kernel_size=(3,3), 
                 padding="same",
                 activation="relu"))
model.add(MaxPool2D(pool_size=(34,34),strides=(34,34)))
model.add(Flatten())
model.add(Dense(units=64,activation="relu",kernel_regularizer='l2'))
model.add(Dense(units=32,activation="relu",kernel_regularizer='l2'))
model.add(Dense(len(breed_name), activation = "softmax"))
model.compile(loss = 'categorical_crossentropy', 
              optimizer = Adam(0.001),
              metrics=['accuracy'])

model.summary()

X_train_and_val, X_test, Y_train_and_val, Y_test = train_test_split(X_data, Y_data, test_size = 0.1,random_state=69)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_and_val, Y_train_and_val, test_size = 0.1
                                                  ,random_state=69)


# Training the model
epochs = 50
batch_size = 128

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
                    validation_data = (X_val, Y_val))



# Plot the training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()


Y_pred = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')


plt.imshow(X_test[1,:,:,:])
plt.show()

# Finding max value from predition list and comaparing original value vs predicted
# print("Originally : ",labels['breed'][np.argmax(Y_test[1])])
# print("Predicted : ",labels['breed'][np.argmax(Y_pred[1])])
print("Originally : ",breed_name[np.argmax(Y_test[1])])
print("Predicted : ",breed_name[np.argmax(Y_pred[1])])

model_jason = model.to_json()
modelfile = 'model.json'
weightfile = 'weight.h5'
with open(modelfile, 'w') as json_file:
    json_file.write(model_jason)
model.save_weights(weightfile)
