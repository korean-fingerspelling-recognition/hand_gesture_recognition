path = './new_dataset/'
letter = {
    'a':0,
    'b':1,
    'c':2,
    'd':3,
    'e':4,
    'f':5,
    'g':6,
}

image_size = 64
num_classes = len(letter)
dropout = 0.5
display_step = 1
learning_rate = 0.0001

# Number of parameters for each layer
layer1 = 32
layer2 = 64
fc1 = 1024
saved_path = './Saved/'

weirdlayershape = int(image_size*image_size/16)*layer2

num_steps = 5500
batch_size = 64
dim = 3
