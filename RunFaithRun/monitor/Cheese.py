'''
    A module used to take screenshots
'''

# import pyscreenshot as pysrc
import pyautogui
import threading


class Cheese(object):
    def __init__(self, region=None, resize=(300, 180)):
        self.region = region
        self.resize = resize

    def cheese(self):
        img = pyautogui.screenshot(region=self.region).resize(self.resize)
        return img


class CheeseThread(threading.Thread):
    def __init__(self, exit_signal, callback=None, **kwargs):
        threading.Thread.__init__(self)
        self.exit_signal = exit_signal
        self.callback = callback
        self.cheese = Cheese(**kwargs)

    def run(self):
        while not self.exit_signal():
            img = self.cheese.cheese()
            if self.callback:
                self.callback(img)
