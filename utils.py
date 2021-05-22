import os
import sys

def min2day(minutes):
    return minutes / 60. / 24

def hour2day(hours):
    return hours / 24.

def min2hour(minutes):
    return minutes / 60.

def day2min(days):
    return days * 24 * 60

def day2hour(days):
    return days * 24

def hour2min(hours):
    return hours * 60
  
    
class stdoutSwitcher:
    def __init__(self):
        self.sysout = sys.stdout
        self.devnull = open(os.devnull, "w")
        self.status = "on"

    def switch(self, choice):
        if choice == "off":
            sys.stdout = self.devnull
            self.status = "off"
        elif choice == "on":
            sys.stdout = self.sysout
            self.status = "on"


def remove_tree(path, info=True):
    if os.path.exists(path):
        if info:
            print(f"Deleting directory '{path}'...")
        if os.path.isdir(path):
            for fname in os.listdir(path):
                if os.path.isdir(path + '/' + fname):
                    remove_tree(path + '/' + fname, info=True)
                else:
                    # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                    open(path + '/' + fname, 'w').close()  # to avoid bug in colab
                    os.remove(path + '/' + fname)
            os.rmdir(path)
        else:
            open(path, 'w').close()  # to avoid bug in colab
            os.remove(path)
    elif info:
        print(f"Directory does not exist yet '{path}'")


def make_dir(path, info=True):
    path_split = path.split('/')
    path_parts = ['/'.join(path_split[:i+1]) for i in range(len(path_split))]
    for path_part in path_parts:
        if not os.path.exists(path_part):
            if info:
                print(f"Making directory '{path_part}'")
            os.mkdir(path_part)