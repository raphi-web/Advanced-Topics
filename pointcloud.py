from typing import Tuple

from my_geometries import Point
import numpy as np
from scipy.spatial import KDTree


class Pointcloud:
    """
    Represents a collection of 2D Points
    """
    def __init__(self, x, y):
        self.__points = np.array([x, y]).T
        self.__kdtree = KDTree(self.__points)

    def __len__(self):
        return self.__points.shape[0]

    def __getitem__(self, index):
        return Point(self.__points[index])

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):

        if self.__index < self.__points.shape[0]:
            value = Point(self.__points[self.__index])
            self.__index += 1
            return value

        raise StopIteration

    @property
    def points(self):
        return [p for p in self]

    @property
    def kdtree(self):
        return self.__kdtree

    def as_array(self):
        return self.__points

    def len(self):
        return self.__points.shape[0]

    def bounds(self):
        """
        x-min, x-max, y-min,y-max
        :return: tuple[float,float,float,float]
        """
        xmin = self.__points[:, 0].min()
        xmax = self.__points[:, 0].max()

        ymin = self.__points[:, 1].min()
        ymax = self.__points[:, 1].max()
        return xmin, xmax, ymin, ymax

    def closest_pair(self) -> Tuple[float, Tuple[Point,Point]]:
        """
        find the closest pair of points in point cloud using divide and conquer
        https://www.geeksforgeeks.org/closest-pair-of-points-using-divide-and-conquer-algorithm/
        :return:
        """
        # sort by x
        points = sorted([p for p in self], key=lambda p: p.x)
        return self.__closest_pair(points)

    def __closest_pair(self, points) -> Tuple[float, Tuple[Point,Point]]:

        if len(points) <= 3:
            return self.__closest_pair_bf(points)

        split_idx = len(points) // 2
        left_half = points[:split_idx]
        right_half = points[split_idx:]

        left_min_with_pair = self.__closest_pair(left_half)
        right_min_with_pair = self.__closest_pair(right_half)

        min_distance, point_pair = min(
            left_min_with_pair, right_min_with_pair, key=lambda dp: dp[0])
        middle = []
        for point in points:
            if abs(point.x - points[split_idx].x) < min_distance:
                middle.append(point)

        middle_min_with_pair = self.__slide_window(
            sorted(middle, key=lambda p: p.y), min_distance)
        if middle_min_with_pair[0] < min_distance:
            min_distance, point_pair = middle_min_with_pair

        return min_distance, point_pair

    def k_nearest_neighbour_kdtree(self, other_pointcloud):
        """
        uses a kdtree to find the k-nearest neighbour between two point clouds
        :param other_pointcloud: Pointcloud B
        :return: Point
        """
        tree = other_pointcloud.kdtree
        _, indexes = tree.query(self.as_array())
        nearest_points = [other_pointcloud[i] for i in indexes]

        return nearest_points

    def k_nearest_neighbour_bf(self, other_pointcloud):
        """
        uses brutal-force to find the k-nearest neighbour
        :param other_pointcloud: Pointcloud B
        :return:
        """
        other_points = other_pointcloud.points
        result = []
        for point in self:
            nearest = min(other_points, key=lambda p: point.distance(p))
            result.append(nearest)

        return result

    @staticmethod
    def gen_random_points(n:float, xmin:float, xmax:float, ymin:float, ymax:float):
        """
        randomly generates points
        :param n: number of points
        :param xmin: minimum X value of points
        :param xmax: maximum X value of points
        :param ymin: minimum Y value of points
        :param ymax: maximum Y value of points
        :return: Pointcloud
        """
        x = np.random.random(size=int(n)) * (xmax - xmin) + xmin
        y = np.random.random(size=int(n)) * (ymax - ymin) + ymin

        return Pointcloud(x, y)

    @staticmethod
    def __closest_pair_bf(points):
        """
        implementation of k-nearest neighbour bf
        :param points:
        :return:
        """
        distances = []
        for pa in points:
            distances_pa = []
            for pb in points:
                if pb != pa:
                    distances_pa.append((pa.distance(pb), (pa, pb)))

            distances.append(min(distances_pa, key=lambda p: p[0]))

        return min(distances, key=lambda p: p[0])

    @staticmethod
    def __slide_window(points, distance):
        """
        the sliding window used in the divide and conquer impl. of closest pair of points
        :param points: list of Points
        :param distance: minimum Distance
        :return: new minimum Distance, the pair of closest Points
        """
        # compare the
        window_size = 7
        length = len(points)
        point_pair = ()
        for i in range(len(points), length):
            for j in range(i+1, min(i+window_size, length)):
                distance_window = points[i].distance(points[j])
                if distance_window < distance:
                    distance = min(distance, distance_window)
                    point_pair = (points[i], points[j])

        return (distance, point_pair)
