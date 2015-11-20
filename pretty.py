from termcolor import colored

def section(msg):
    print(colored("\n::", "blue", attrs=["bold"]), colored(msg, attrs=["bold"]))
def task(msg):
    print(colored("==>", "green", attrs=["bold"]), colored(msg, attrs=["bold"]))
def subtask(msg):
    print(colored(" ->", "blue", attrs=["bold"]), colored(msg, attrs=["bold"]))

from progressbar import Bar, SimpleProgress, Percentage, ProgressBar, Timer, AbsoluteETA

def progress(number, **kwargs):
    return ProgressBar(max_value=number, widgets=[Percentage(), ' (', SimpleProgress(), ') ', Bar(), ' ', Timer(), ' ', AbsoluteETA()], **kwargs).start()
