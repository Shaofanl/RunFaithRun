from time import sleep


def countdown(n):
    for i in range(n)[::-1]:
        print(i)
        sleep(1)
