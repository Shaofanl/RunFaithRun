from RunFaithRun.learning.DQL import DDQL, Environment
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from keras.layers import Dense, Flatten, Input, Dropout
from keras.layers import Conv2D, MaxPooling2D
import pyautogui

from RunFaithRun.control.win import PressKey, ReleaseKey, MoveMouse
from RunFaithRun.monitor.speed import Speedometer
from time import sleep
from RunFaithRun.utils import countdown

class MEC(Environment):
    def __init__(self, region=(0, 40, 960, 540), resize=(296, 176)):
        self.region = region
        self.resize = resize
        self.state_shape = (None,)+resize[::-1]+(3,)
        self.action_count = 12  # keyboard, mouse, space, shift
        self.max_steps = 200
        self.speedometer = Speedometer()

        self.action_names = ['move_forward', 'move_backward', 'move_left', 'move_right',
                            'look_right', 'look_down', 'look_left', 'look_up',
                            'long_jump', 'short_jump', 'shift', 'turn']
        self.failed_flag = False 

    def Qnetwork(self, state):
        x = state

        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)

        x = Dropout(0.5)(x)
        x = Dense(self.action_count, activation='sigmoid')(x)
        return x

    @property
    def state(self):
        img = pyautogui.screenshot(region=self.region).resize(self.resize)
        img = np.array(img)
        self._state_buf = img
        return img 

    def failed(self):
        return np.all(self._state_buf.mean(0).mean(0) > 240)

    def successful(self):
        return self.steps >= self.max_steps
    
    def done(self):
        if self.failed():
            self.failed_flag = True
            return True
        elif self.successful():
            return True
        return False
    
    def step(self, action):
        # dx=-1, dx=1, dy=-1, dy=1
        self.steps += 1

        def press(hexkey, delay=0.2):
            PressKey(hexkey)  # W 
            sleep(delay)
            ReleaseKey(hexkey)

        # keyboard, mouse, space, shift
        from RunFaithRun.control.win import PressKey, ReleaseKey, MoveMouse
        if action == 0:
            press(0x11)  # W
        elif action == 1:
            press(0x1F)  # S
        elif action == 2:
            press(0x1E)  # A
        elif action == 3:
            press(0x20)  # D
        elif action == 4:
            MoveMouse(100, 0)
        elif action == 5:
            MoveMouse(0, 100)
        elif action == 6:
            MoveMouse(-100, 0)
        elif action == 7:
            MoveMouse(0, -100)
        elif action == 8:
            press(0x39, delay=0.5)  # space
        elif action == 9:
            press(0x39, delay=0.1)  # space
        elif action == 10:
            press(0x2A)  # shift
        elif action == 11:
            press(0x10)  # Q 
                                

        if self.failed():
            return -1000
        elif self.successful():
            return 0 
        else:
            return \
              np.sum((self.get_speed()**2)*[1, 1e-2, 1])*10

    def get_speed(self):
        return self.speedometer.get_speed()

    def reset(self):
        self.steps = 0
        self.speedometer.reset()
        if self.failed_flag:
            countdown(11)
        self.failed_flag = False


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.__enter__()

    mec = MEC()

#   mec.reset()
#   while True:
#       mec.state
#       for i in range(10):
#           print(i)
#           mec.step(i)
#           sleep(1)

#   mec.reset()
#   while True:
#       print(mec.get_speed())
#       sleep(1)

    ddql = DDQL(env=mec)
    ddql.build(learning_rate=1e-4)  #, gamma=0.9)
    ddql.train(
            continued=True,
            max_replays=500,
            iterations=np.inf,
            epsilon=0.1, epsilon_decay=1-5e-4, epsilon_min=0.1)

