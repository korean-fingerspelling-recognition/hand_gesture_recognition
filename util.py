import tensorflow as tf
import os
import numpy as np
from PIL import ImageOps
import Image
import cv2
import parameters

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
	
   imgFiles = []
   for letter in allFiles:
      for image in os.listdir(os.path.join(FLAGS.data_path, letter)):
         if image.endswith('.jpg') and not image.startswith('._'):
	    absPath = os.path.join(FLAGS.data_path, letter, image)
	    imgFiles.append(absPath)

   idx = np.random.permutation(len(imgFiles))
   idx = idx[0:batch_size]

   images = []
   labels = []
   mask_images = []

   for item in idx:
   # Get Image
	img = Image.open(imgFiles[item])
	img = np.array(img.resize((image_size, image_size)))
      	images.append(img)
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

   return images.astype(np.float32), labels



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
