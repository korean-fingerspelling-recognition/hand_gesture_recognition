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

- image_size는 작을 수록 training이 잘되는 경향을 보였다.
- append_handmask 쓰기 위해서는 dim 4 필요
- number of steps가 5천 이후 부터는 크게 변화가 없다.
- dataset은 각 자/모음 당 filename에 그 자모음을 적어서 파일 밑에 해당 이미지 저장. 저장된 파일들을 하나의 파일 results 밑에 넣고 main.py와 같은 곳에 파일 두면 알아서 적용이 된다. 혹은 --data_path로 변경 가능
- save_steps는 해당 컴퓨터의 용량에 맞게 변경하면 된다.
