import numpy as np
from numba import float32


class Point():

    def __init__(self, xy):
        self.__x = xy[0]
        self.__y = xy[1]

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y
    

    def distance(self, other):
        a = self.x - other.x
        b = self.y - other.y
        return np.sqrt(a**2 + b**2)

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other):
        return (self.x == other.x) & (self.y == other.y)

