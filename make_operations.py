import os
from os.path import dirname, basename, isfile, join
def make_operations():
    modules = os.listdir('./Operations Folder')
    test = open('Operations.py','w+')
    functions = []
    for module in modules:
        if module[-3:] != '.py':
            continue
        if module == '__init__.py':
            continue
        f = open('./Operations Folder/' + module,'r')
        test.write(f.read())
        test.write('\n')
        functions.append(module[:-3])
    return(functions)
def make_otherfunctions():
    modules = os.listdir('./PeripheryFunctions')
    test = open('Periphery.py','w+')
    functions = []
    for module in modules:
        if module[-3:] != '.py':
            continue
        if module == '__init__.py':
            continue
        f = open('./PeripheryFunctions/' + module,'r')
        test.write(f.read())
        test.write('\n')
        functions.append(module[:-3])
    return(functions)
