from pynput.keyboard import Key, Controller
from pynput.mouse import Controller
import time

keyboard = Controller()
mouse = Controller()

# Press and release space
time.sleep(2)
while True:
    # print('press')
    # keyboard.press(Key.space)
    # time.sleep(0.5)
    # keyboard.release(Key.space)

    print('move')
    mouse.move(-10, 10)

    time.sleep(0.1)

# cannot work in ME 
