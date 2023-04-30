import numpy as np
from my_geometries import Point
from pointcloud import Pointcloud
"""
just various tests to verify the implementation of pointcloud
"""

def test_pointcloud_len():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    pc = Pointcloud(x, y)
    assert len(pc) == 3


def test_pointcloud_getitem():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    pc = Pointcloud(x, y)
    assert pc[0] == Point([1, 4])
    assert pc[1] == Point([2, 5])
    assert pc[2] == Point([3, 6])


def test_pointcloud_iter():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    pc = Pointcloud(x, y)
    expected_points = [Point([1, 4]), Point([2, 5]), Point([3, 6])]
    for i, point in enumerate(pc):
        assert point == expected_points[i]


def test_pointcloud_len_method():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    pc = Pointcloud(x, y)
    assert pc.len() == 3


def test_pointcloud_bounds():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    pc = Pointcloud(x, y)
    assert pc.bounds() == (1, 3, 4, 6)


def test_pointcloud_closest_pair():
    x = np.array([0, 0.5, 1])
    y = np.array([0, 0.5, 1])
    pc = Pointcloud(x, y)
    expected_min_distance = 0.5 * np.sqrt(2)
    _, point_pair = pc.closest_pair()
    assert point_pair[0] == Point([0, 0])
    assert point_pair[1] == Point([0.5, 0.5])


def test_point_cloud_k_nearest_neighbour_bf():
    xa = [0, 1, 2]
    ya = [0, 1, 2]

    xb = [0, 1, 2]
    yb = [0, 1.2, 2.2]
    pointcloud_a = Pointcloud(xa, ya)
    pointcloud_b = Pointcloud(xb, yb)
    nearest = pointcloud_a.k_nearest_neighbour_bf(pointcloud_b)
    nearest_pairs = list(zip(pointcloud_a, nearest))

    a = [Point(i) for i in zip(xa, ya)]
    b = [Point(i) for i in zip(xb, yb)]
    expected = list(zip(a, b))

    for (exp, res) in zip(expected, nearest_pairs):
        assert (exp[0] == res[0]) & (exp[1] == res[1])


def test_point_cloud_k_nearest_neighbour_kdtree():
    xa = [0, 1, 2]
    ya = [0, 1, 2]

    xb = [0, 1, 2]
    yb = [0, 1.2, 2.2]

    a = [Point(i) for i in zip(xa, ya)]
    b = [Point(i) for i in zip(xb, yb)]
    expected = zip(a, b)

    pointcloud_a = Pointcloud(xa, ya)
    pointcloud_b = Pointcloud(xb, yb)

    nearest = pointcloud_a.k_nearest_neighbour_kdtree(pointcloud_b)
    result = zip(pointcloud_a.points, nearest)

    expected = list(zip(a, b))
    for (exp, res) in zip(expected, result):
        assert (exp[0] == res[0]) & (exp[1] == res[1])


if __name__ == "__main__":
    test_pointcloud_len()
    test_pointcloud_iter()
    test_pointcloud_bounds()
    test_pointcloud_getitem()
    test_pointcloud_len_method()
    test_pointcloud_closest_pair()
    test_point_cloud_k_nearest_neighbour_bf()
    test_point_cloud_k_nearest_neighbour_kdtree()
