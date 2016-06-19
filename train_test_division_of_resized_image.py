import os
from os import listdir
import csv
from os.path import isfile,join
from PIL import Image
import random

category_dict = {"1":"ns","2":"ew","3":"flat","4":"other"}

image_name_mark_dict = dict()

row_count = 0
with open("id_train.csv",'rU') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		row_count += 1
		if row_count != 1:
			#if row_count > 10:
			# 	break			
			image_name = (row[0]).split(",")[0]
			image_mark = (row[0]).split(",")[1]
			image_name_mark_dict[image_name] = image_mark
			print image_name,image_mark

print image_name_mark_dict

train_and_val_image_names = image_name_mark_dict.keys()
print train_and_val_image_names

random.seed(410)
random.shuffle(train_and_val_image_names)
print train_and_val_image_names

print len(train_and_val_image_names)


output_file = open("/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test_imagename_mark.txt","w+")
import shutil
loop_count = 0
for file_names in train_and_val_image_names:
	loop_count += 1
	print loop_count
	if loop_count < 7105:
		print "skipping this"
		# label = image_name_mark_dict[file_names]
		# folder = category_dict[label]
		# src = "/Users/ROBIN/Desktop/dsg/resized_to_32/"+file_names+".jpg"
		# dest = "/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/train"+"/"+folder+"/"+file_names+".jpg"
		# shutil.copy(src,dest)
	else :
		label = image_name_mark_dict[file_names]
		folder = category_dict[label]
		src = "/Users/ROBIN/Desktop/dsg/resized_to_32/"+file_names+".jpg"
		#dest = "/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test/"+folder+"/"+file_names+".jpg"
		dest = "/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test/"+file_names+".jpg"
		shutil.copy(src,dest)
		output_line = (str(file_names)+","+str(label)+"\n")
		output_file.writelines(output_line)


#prepare test data set
"""
import shutil
row_count = 0
test_images = list()
with open("sample_submission4.csv",'rU') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		row_count += 1
		if row_count != 1:
			# if row_count > 10:
			 	# break			
			image_name = (row[0]).split(",")[0]
			test_images.append(image_name)

loop_count = 0
for file_names in test_images:
	loop_count += 1
	src = "/Users/ROBIN/Desktop/dsg/resized_to_32/"+file_names+".jpg"
	dest = "/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/test/"+file_names+".jpg"
	shutil.copy(src,dest)
	print loop_count
"""
