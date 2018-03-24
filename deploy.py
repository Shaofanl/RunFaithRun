import keras
import pyautogui
import numpy as np
import tensorflow as tf
from RunFaithRun.control.win import PressKey, ReleaseKey


def print_progress_bar(value, max_value=1.0, prefix='', length=100):
    left_len = length - len(prefix)
    act_len = int(value/max_value * left_len)
    space_len = left_len-act_len
    print ('{}{}{}'.format(prefix, '#'*act_len, '-'*space_len),
            # end='\r',
            flush=True)

if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.__enter__()

    model = keras.models.load_model('./models/model.h')
    window_size = 3
    threshold = 0.5

    inputs = []
    pressed = False
    while True:
        img = pyautogui.screenshot(region=(0, 40, 960, 540)).resize((296, 176))
        # img = pyautogui.screenshot().resize((300, 180))
        # img = pyautogui.screenshot().resize((296, 176))
        img = np.array(img)
        # img = np.expand_dims(img, 0)

        if len(inputs) >= window_size:
            inputs.pop(0)
            inputs.append(img)
            # v = np.concatenate(inputs, -1)
            v = np.array(inputs)
            v = v.transpose(0, 3, 1, 2).reshape(1, window_size*3, v.shape[1], v.shape[2]).transpose(0, 2, 3, 1) 
            jump_confidence = model.predict(v)[0]
            # print( jump_confidence )
            print_progress_bar(value=jump_confidence[0], prefix='Jump confidence: ')

            if jump_confidence > threshold:
                # print('>>>>>>>>>>>> jump <<<<<<<<<<<<<<<')
                print_progress_bar(value=jump_confidence[0], prefix='Jump Faith Jump! ')
                if pressed == False:
                    # pyautogui.press('space')
                    PressKey(0x39)
                    pressed = True
            else:
                if pressed == True:
                    # pyautogui.keyUp('space')
                    ReleaseKey(0x39)
                    pressed = False
        else:
            inputs.append(img)

