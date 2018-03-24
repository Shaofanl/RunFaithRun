from pynput import keyboard
from pynput import mouse
# import pyscreenshot as pysrc
from time import time
import os

from RunFaithRun.monitor.Cheese import CheeseThread
from RunFaithRun.utils import countdown


def exit_signal():
    global stop_flag
    return stop_flag


def callback_gen(log_dir):
    def callback(img):
        timetag = time()
        img.save(log_dir+'/'+str(timetag)+'.jpg')
    return callback


def on_move(x, y):
    print('Pointer moved to {0}'.format((x, y)))
    global stop_flag
    if stop_flag:
        return False


def on_press(key):
    timetag = time()
    if key == keyboard.Key.space:
        logging.info('space pressed:[{}]'.format(timetag))
    elif key == keyboard.Key.shift:
        logging.info('shift pressed:[{}]'.format(timetag))
    else:
        if hasattr(key, 'char'):
            if key.char != 'w':
                logging.info('{} pressed:[{}]'.format(key.char, timetag))
        else:
            logging.info('{} pressed:[{}]'.format(key, timetag))

    # if hasattr(key, 'char'):
    #     print('alphanumeric key {0} pressed'.format( key.char))
    #     logging.info(key.char+' pressed:[{}]'.format(timetag))
    # else:
    #     print(key==keyboard.Key.space)
    #     print(key)
    # else:
    # except AttributeError:
    #   print('special key {0} pressed'.format( key))


def on_release(key):
    timetag = time()
    global stop_flag
    # print('{0} released'.format(key))
    # if hasattr(key, 'char'):
    #     logging.info(key.char+' released:[{}]'.format(timetag))

    if key == keyboard.Key.space:
        logging.info('space release:[{}]'.format(timetag))
    elif key == keyboard.Key.shift:
        logging.info('shift release:[{}]'.format(timetag))
    elif key == keyboard.Key.esc:  # Stop listener
        stop_flag = True
        return False


if __name__ == '__main__':
    countdown(5)

    log_dir = './data/'+str(time())
    os.makedirs(log_dir)
    print(log_dir)

    import logging
    logging.basicConfig(
            filename=log_dir+'/log.txt',
            level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    global stop_flag
    stop_flag = False

    cheese_list = [CheeseThread(
        exit_signal=exit_signal,
        callback=callback_gen(log_dir)) for _ in range(1)]
    k_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
    m_listener = mouse.Listener(on_move=on_move)

    k_listener.__enter__()
    for cheese in cheese_list:
        cheese.start()
    # m_listener.__enter__()

    k_listener.join()
    for cheese in cheese_list:
        cheese.join()
    # m_listener.join()
