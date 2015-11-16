from termcolor import colored

def section(msg):
    print(colored("\n::", "blue", attrs=["bold"]), colored(msg, attrs=["bold"]))
def task(msg):
    print(colored("==>", "green", attrs=["bold"]), colored(msg, attrs=["bold"]))
def subtask(msg):
    print(colored(" ->", "blue", attrs=["bold"]), colored(msg, attrs=["bold"]))

