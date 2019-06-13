
import tensorflow as tf
import os
import numpy as np
from PIL import ImageOps
import Image
import cv2
import random

# get batch of images
def get_next_batch(batch_size, image_size):
   flags = tf.app.flags
   FLAGS = flags.FLAGS

   allFiles = os.listdir(FLAGS.data_path)
 
   # make a dict containing labels
   _letter = dict()
   for i in range(len(allFiles)):
	_letter.update({str(allFiles[i]):i})
   num_classes = len(allFiles)    
	
   imgFiles = [] # train image file
   t_imgFiles = [] # test image file
   for letter in allFiles:
	img_path = os.path.join(FLAGS.data_path, letter)
	all_img_list = os.listdir(img_path)
	train_imgs = int(len(all_img_list) * 0.9)
	for i in range(len(all_img_list)):
		image = all_img_list[i]
		if image.endswith('.jpg') and not image.startswith('._'):
			absPath = os.path.join(img_path, image)
			if i > train_imgs: t_imgFiles.append(absPath)
			else: imgFiles.append(absPath)


   #### train image ####
   idx = np.random.permutation(len(imgFiles))
   idx = idx[0:batch_size]

   images = []
   labels = []
   mask_images = []

   for item in idx:
   # Get Image
	img = Image.open(imgFiles[item])
	img = np.array(img.resize((image_size, image_size)))
	
	light_img = light_effect(imgFiles[item])
	if random.randint(0, 10) < 5: images.append(img)
	else: images.append(light_img)

	convert = convert_img(imgFiles[item])
	mask_images.append(convert)
	# Get Labels
        labels.append(int(_letter[imgFiles[item].split('/')[-2]]))

   images = np.reshape(images, [-1, image_size, image_size, 3])
   mask_images = np.reshape(mask_images, [-1, image_size, image_size, 1])
   labels = np.array(labels)
   labels = np.reshape(labels,[-1,1])

   if FLAGS.append_handmask:
	images = np.concatenate((images, mask_images), axis = 3)   

   #### test image #####
   test_img_num = len(t_imgFiles)
   t_idx = np.random.permutation(len(t_imgFiles))
   t_idx = t_idx[0:test_img_num]

   t_images = []
   t_labels = []
   t_mask_images = []

   for t_item in t_idx:
   # Get Image
	img = Image.open(t_imgFiles[t_item])
	img = np.array(img.resize((image_size, image_size))
	t_images.append(img)
	convert = convert_img(t_imgFiles[t_item])
	t_mask_images.append(convert)
	# Get Labels
        t_labels.append(int(_letter[t_imgFiles[t_item].split('/')[-2]]))

   t_images = np.reshape(t_images, [-1, image_size, image_size, 3])
   t_mask_images = np.reshape(t_mask_images, [-1, image_size, image_size, 1])
   t_labels = np.array(t_labels)
   t_labels = np.reshape(t_labels,[-1,1])

   if FLAGS.append_handmask:
	t_images = np.concatenate((t_images, t_mask_images), axis = 3)   

   return images.astype(np.float32), labels, t_images.astype(np.float32), t_labels



def convert_img(path):
    
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (FLAGS.image_size, FLAGS.image_size))
    convert_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_boundary = np.array([0, 40, 30], dtype="uint8")
    high_boundary = np.array([43, 255, 254], dtype = "uint8")

    skinmask = cv2.inRange(convert_img, low_boundary, high_boundary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skinmask = cv2.erode(skinmask, kernel, iterations = 2)
    skinmask = cv2.dilate(skinmask, kernel, iterations = 2)

    low_boundary_2 = np.array([170, 80, 30], dtype="uint8")
    high_boundary_2 = np.array([180, 255, 250], dtype="uint8")

    skinmask2 = cv2.inRange(convert_img, low_boundary_2, high_boundary_2)
    skinmask = cv2.addWeighted(skinmask, 0.5, skinmask2, 0.5, 0.0)
    skinmask = cv2.medianBlur(skinmask, 5)
    skin = cv2.bitwise_and(frame, frame, mask=skinmask)
    frame = cv2.addWeighted(frame, 1.5, skin, -0.5, 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinmask)
    h, s, v = cv2.split(skin) 
    return v # black except the part with hand

def light_effect(path):
	
	flags = tf.app.flags
	FLAGS = flags.FLAGS

	frame = cv2.imread(path)
	frame = cv2.resize(frame, (FLAGS.image_size, FLAGS.image_size))
	light_effect = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(light_effect)
	append = random.randint(1, 30)
	h = h + append
	h = cv2.inRange(h, 0, 256)
	img = cv2.merge((h, s, v))
	img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
	return img
