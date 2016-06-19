import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import h5py
import numpy as np

img_width, img_height = 32, 32

train_data_dir = '/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/train'
validation_data_dir = '/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/validation'
# test_data_dir = '/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test'
nb_train_samples = 7104
nb_validation_samples = 896
nb_epoch = 20

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


sgd  = keras.optimizers.SGD(lr=0.0625,decay = 1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

##########

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen2 = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,shuffle=True,classes=["ns","ew","flat","other"])

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,shuffle=True,classes=["ns","ew","flat","other"])

# test_generator = test_datagen2.flow_from_directory(
#     test_data_dir,
#     target_size =(img_width, img_height),
#     batch_size = 64,class_mode = None,shuffle=False 
#     )

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples, verbose = 1)

# a = model.evaluate_generator(test_generator,896,1)
# print a

# np.savetxt("PredProb_validation_set_as_test.csv",a,delimiter=",")

model.save_weights('weights_test_set_as_array.h5')
# # json_string = model.to_json()

# # model.load_weights('first_try.h5')

########
"""predict on the validation set"""
# test_data_dir = '/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test/val_test'
# output_file = open("/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test_prediction.txt","w+")
# import os
# from os import listdir
# from os.path import isfile,join
# from PIL import Image
# # import Image
# os.chdir(test_data_dir)
# cwd = os.getcwd()
# files_list= [f for f in listdir(cwd) if isfile(join(cwd,f))]
# print files_list

# loop_count = 0
# for file in files_list:
#     print file
#     loop_count += 1
#     # if loop_count >10:
#         # break
#     if file[-4:] != ".jpg":
#         continue
#     image_array = np.asarray(Image.open(file))
#     image_array=np.swapaxes(image_array,0,2)
#     image_array=np.swapaxes(image_array,1,2)
#     # print image_array
#     reshaped = image_array[np.newaxis,...]
#     # print reshaped
#     # print image_array.shape,image_array.size
#     # print reshaped.shape
#     a = model.predict(reshaped,batch_size=1)[0]
#     output_line = ((str(file[:-4])+","+str(a[0])+","+str(a[1])+","+str(a[2])+","+str(a[3]))+"\n")
#     output_file.writelines(output_line)
#     print a 

"""predict on the test set"""
test_data_dir = '/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/test/test'
output_file = open("/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/test/test_prediction.txt","w+")
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
    # if loop_count >10:
        # break
    if file[-4:] != ".jpg":
        continue
    image_array = np.asarray(Image.open(file))
    image_array=np.swapaxes(image_array,0,2)
    image_array=np.swapaxes(image_array,1,2)
    reshaped = image_array[np.newaxis,...]
    a = model.predict(reshaped,batch_size=1)[0]
    output_line = ((str(file[:-4])+","+str(a[0])+","+str(a[1])+","+str(a[2])+","+str(a[3]))+"\n")
    output_file.writelines(output_line)
    print a 

