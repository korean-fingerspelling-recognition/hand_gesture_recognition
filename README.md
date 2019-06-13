# hand_gesture_recognition
## flags information

```
optional arguments:
  -h, --help            show this help message and exit
  --append_handmask [APPEND_HANDMASK]
                        If true, train with handmask appended
  --image_size IMAGE_SIZE
                        define input image size
  --dropout DROPOUT     dropout rate
  --display_step DISPLAY_STEP
                        define display_step
  --use_bottleneck [USE_BOTTLENECK]
                        change the sub-layer on Resnet
  --saved_path SAVED_PATH
                        path to save model: Will create automatically if it
                        does not exist
  --num_steps NUM_STEPS
                        define the number of steps to train
  --train_mode [TRAIN_MODE]
                        True if in train mode, False if in Test mode
  --data_path DATA_PATH
                        path of the dataset: inside this directory there
                        should be files named with label
  --learning_rate LEARNING_RATE
                        learning_rate during training
  --batch_size BATCH_SIZE
                        batch size during training
  --save_steps SAVE_STEPS
                        number of save steps
  --dim DIM             input_channel is 4
  --num_classes NUM_CLASSES
                        number of classes
  --model_num MODEL_NUM
                        0: default model / 1: resnet / 2: bottlenect resnet
  --gpu                 choose between gpu 0 and gpu 1
  ```

