import os
import csv
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import h5py
import numpy as np

os.chdir("/Users/ROBIN/Desktop/dsg/")
cwd = os.getcwd()

img_width, img_height = 32, 32
model = Sequential()
model.add(Convolution2D(32, 4, 4, input_shape=(3, img_width,img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 4, 4))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 4, 4))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))
model.load_weights('my_model_weights_421.h5')
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

# test_data_dir = '/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/test'
# test_datagen2 = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen2.flow_from_directory(
#     test_data_dir,
#     target_size =(img_width, img_height),
#     batch_size = 32,class_mode = None,shuffle=False 
#     )

# a = model.predict_generator(test_generator,13999,1)
# print a
# np.savetxt("predicted_probabilities_.csv",a,delimiter=",")

test_data_dir = '/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test/val_test'
output_file = open("/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test_prediction.txt","w+")
import os
from os import listdir
from os.path import isfile,join
from PIL import Image
# import Image
os.chdir(test_data_dir)
cwd = os.getcwd()
files_list= [f for f in listdir(cwd) if isfile(join(cwd,f))]
print files_list

loop_count = 0
for file in files_list:
    print file
    loop_count += 1
    if loop_count >10:
        break
    if file[-4:] != ".jpg":
        continue
    image_array = np.asarray(Image.open(file))
    image_array=np.swapaxes(image_array,0,2)
    image_array=np.swapaxes(image_array,1,2)
    # print image_array
    reshaped = image_array[np.newaxis,...]
    # print reshaped
    # print image_array.shape,image_array.size
    # print reshaped.shape
    a = model.predict(reshaped,batch_size=1)[0]
    output_line = ((str(file[:-4])+","+str(a[0])+","+str(a[1])+","+str(a[2])+","+str(a[3]))+"\n")
    output_file.writelines(output_line)
    print a 


