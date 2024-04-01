import pandas as pd
import numpy as np
import math


class Point:
    def __init__(self, pt, x, y):
        self.ptid = pt
        self.x = x
        self.y = y


class AngleObs:
    """
    This class is an angle observation class. It contains three points (from, at, to),
    and the corresponding angle measurement in different formats.
    This class always takes dms format as input and convert to deg or rad
    """
    def __init__(self, from_pt, at_pt, to_pt, angle):
        self.from_pt = from_pt
        self.at_pt = at_pt
        self.to_pt = to_pt
        self.angle_dms = angle
        self.angle_d = self.angle_dms[0] + self.angle_dms[1]/60 + self.angle_dms[2] / 3600
        self.angle_rad = self.angle_d * math.pi / 180

    def __dms2deg__(self):
        """convert angle observation from [deg min sec] to deg"""
        self.angle_d = self.angle_dms[0] + self.angle_dms[1]/60 + self.angle_dms[2]/3600

    def __dms2rad__(self):
        """convert angle observation from [deg min sec] to rad"""
        self.angle_rad = self.angle_d * math.pi / 180


class DistObs:
    """
    This class is a distance observation class. It contains two points (from, to),
    and the corresponding distance measurement
    """
    def __init__(self, from_pt, to_pt, distance):
        self.from_pt = from_pt
        self.to_pt = to_pt
        self.dist = distance