import numpy as np
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arr = np.array([[1], [1], [2],[2]])
    arr1 = arr[arr==1].reshape(2,)
    arr2 = arr[arr==2].reshape(1,)
    print(arr1.shape,arr2.shape)
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
