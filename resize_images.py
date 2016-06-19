import os
from os import listdir
from os.path import isfile,join

#read image files
import numpy as np
os.chdir("/Users/ROBIN/Desktop/dsg/roof_images")
cwd = os.getcwd()
files_list= [f for f in listdir(cwd) if isfile(join(cwd,f))]
print files_list
loop_count = 0
from PIL import Image
from resizeimage import resizeimage
width_list = list()
height_list = list()
aspect_ratio_list = list()

for image_file in files_list:
	loop_count += 1
	#if loop_count > 10:
	# 	break
	print loop_count
	im = Image.open(image_file)
	width,height = im.size	
	aspect_ratio = float(width)/float(height)
	width_list.append(width)
	height_list.append(height)
	aspect_ratio_list.append(aspect_ratio)

	desired_pixels = 32
	image_size = desired_pixels*desired_pixels
	# resized_image = resizeimage.resize_contain(im, [desired_pixels,desired_pixels],Image.BILINEAR)
	resized_image = im.resize((desired_pixels,desired_pixels),Image.BILINEAR)
	os.chdir("/Users/ROBIN/Desktop/dsg/resized_to_32")
	cwd = os.getcwd()
	# print cwd
	resized_image.save(image_file)

	new_width,new_height = resized_image.size

	print width,height,new_width,new_height
	os.chdir("/Users/ROBIN/Desktop/dsg/roof_images")


print "number of image files read",loop_count
print "minimum width of any image in the list is ",min(width_list)
print "maximum width of any image in the list is",max(width_list)
print "minimum height of any image in the list is",min(height_list)
print "maximum height of any image in the list is",max(height_list)

print "percentiles of width values",([np.percentile(width_list,10*i) for i in range(0,11)])
print "percentiles of height values",([np.percentile(height_list,10*i) for i in range(0,11)])
print "percentiles of aspect ratios",([np.percentile(aspect_ratio_list,10*i) for i in range(0,11)])



# #read the train csv file
# os.chdir("/Users/ROBIN/Desktop/dsg/")
# import csv
# row_count = 0
# with open("id_train.csv",'rU') as csvfile:
# 	spamreader = csv.reader(csvfile, delimiter=',')
# 	for row in spamreader:
# 		print [(row[0]).split(",")]
# 		row_count += 1