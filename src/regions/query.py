from enum import Enum


class Query(Enum):
    POINT = 0
    RAY = 1
    PLANE = 2
    BOX = 3
