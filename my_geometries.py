from typing import Tuple, List

import numpy as np
from scipy.spatial import Delaunay
from shapely import geometry as shapely_geom

class Point3D:
    """
    simple class to define a point in three-dimensional space
    """

    def __init__(self, xyz) -> None:
        self.__x = xyz[0]
        self.__y = xyz[1]
        self.__z = xyz[2]

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def z(self):
        return self.__z

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z])

    def distance(self, other):
        """
        Calculates euclidian distance from one point to another
        :param other: Point, point to calculate the distance to
        :return:
        """
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z

        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


class Polyhedra:
    """
    Describes a Polyhedram a Polygon in 3D-Space
    """

    def __init__(self, vertices) -> None:
        self.__vertices = vertices

    @property
    def vertices(self):
        return self.__vertices

    @property
    def xyz(self):
        return np.stack([p.xyz for p in self.__vertices])

    def point_in_polyhedra(self, point: Point3D) -> bool:
        """
        tests if a point lies within a polyhedra
        using Delaunay triangulation
        :param point: Point3D, the point to test for
        :return: Boolean
        """
        # https://stackoverflow.com/questions/29311682/finding-if-point-is-in-3d-poly-in-python
        return Delaunay(self.xyz).find_simplex(point.xyz) >= 0


class Point:
    """
    Point in 2D-space
    """

    def __init__(self, xy):
        self.__x = xy[0]
        self.__y = xy[1]

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def xy(self) -> 'tuple[float,float]':
        return self.x, self.y

    def distance(self, other) -> float:
        """
        D
        :param other:
        :return:
        """
        a = self.x - other.x
        b = self.y - other.y
        return np.sqrt(a ** 2 + b ** 2)

    def __str__(self) -> str:
        return f"POINT ({self.x} {self.y})"

    def __repr__(self) -> str:
        return f"POINT ({self.x} {self.y})"

    def __eq__(self, other) -> bool:
        # the number of digits in 10e-8
        # should correspond to 32 bit
        dx = abs(self.x - other.x) < 10e-8
        dy = abs(self.y - other.y) < 10e-8
        return dx & dy
        # return (self.x == other.x) & (self.y == other.y)


class Edge():
    def __init__(self, point1: Point, point2: Point) -> None:
        self.__vertices = (point1, point2)

    @property
    def vertices(self) -> 'tuple[Point,Point]':
        return self.__vertices

    @property
    def point_1(self) -> Point:
        return self.__vertices[0]

    @property
    def point_2(self) -> Point:
        return self.__vertices[1]

    @property
    def x(self):
        return self.point_1.x, self.point_2.x 

    @property
    def y(self):
        return self.point_1.y, self.point_2.y

    @property
    def xy(self) -> 'tuple[list[float],list[float]]':
        return [self.point_1.x, self.point_2.x], [self.point_1.y, self.point_2.y]

    def length(self) -> float:
        return self.point_1.distance(self.point_2)

    def __str__(self) -> str:
        (x1, y1) = self.point_1.xy
        (x2, y2) = self.point_2.xy
        return f"LINESTRING ({x1} {y1}, {x2} {y2})"

    def __iter__(self):
        return iter(self.__vertices)

    def intersects(self, other):
        """
        cross product method to check if edges intersects
        https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-edge-segments-intersect
        runs from p -> p+r and second edge q to q+s

        check the points if intersection
        took me a while to figure this out but it gets easier
        if you draw it https://www.geogebra.org/calculator/nucy4npt
        comparing the signs of the differences between the x and y
        coordinates of each pair of endpoints.
        If all differences have the same sign, it means that the edge segments overlap
        and intersect.

        """
        # initial two starting points of the edge
        p = np.array([self.point_1.x, self.point_1.y])
        q = np.array([other.point_1.x, other.point_1.y])

        # the points p1 and q1 can be expressed as p + r and q + s
        p_plus_r = np.array([self.point_2.x, self.point_2.y])
        q_plus_s = np.array([other.point_2.x, other.point_2.y])

        r = p_plus_r - p
        s = q_plus_s - q

        # any point on first edge is p + t * r
        # any point on second edge is q + u * s
        # intersects if we can find t and u such that: p + t*r = q+u*s
        # t = (q - p) × s / (r × s)
        # u = (q - p) × r / (r × s)

        r_cross_s = np.cross(r, s)
        q_min_p_cross_s = np.cross((q - p), s)
        q_min_p_cross_r = np.cross((q - p), r)

        if r_cross_s == 0 and q_min_p_cross_r == 0:
            # collinear
            p_1 = Point(p)
            p_2 = Point(q)  # rename for ease
            q_1 = Point(q_plus_s)
            q_2 = Point(p_plus_r)

            # are points the same
            for pi in [p_1, p_2]:
                for pj in [q_1, q_2]:
                    if pi == pj:
                        return True, pi

            else:
                # edges do not touch but do they overlap?
                # check if x or y values overlap somewhere

                x_vals = sorted([q_1.x, q_2.x, p_1.x, p_2.x])
                y_vals = sorted([q_1.y, q_2.y, p_1.y, p_2.y])
                x_min = x_vals[0]
                x_max = x_vals[-1]
                y_min = y_vals[0]
                y_max = y_vals[-1]
                overlap = (x_min < x_max) and (y_min < y_max)
                if overlap:
                    p_overlap_min = Point([x_vals[1], y_vals[1]])
                    p_overlap_max = Point([x_vals[-2], y_vals[-2]])
                    return True, (p_overlap_min, p_overlap_max)
                else:
                    return False, None

        if r_cross_s == 0 and q_min_p_cross_r != 0:
            # parallel and non-intersecting
            return False, None

        t = q_min_p_cross_s / r_cross_s
        u = q_min_p_cross_r / r_cross_s
        if (r_cross_s != 0) and (0 <= t <= 1) and 0 <= u <= 1:
            # edges intersect at point q + u*s
            return True, Point(q + u * s)

        # non-intersecting and non-parallel
        return False, None


class Polygon:
    def __init__(self, vertices: 'list[Point]') -> None:
        self.__vertices = vertices

    def __len__(self) -> int:
        return len(self.__vertices)

    @property
    def vertices(self) -> 'list[Point]':
        """
        Vertices of Polygon
        :return: list of Points
        """
        return self.__vertices

    @property
    def edges(self) -> 'list[Edge]':
        """
        Edges of the polygon
        :return: List of Edges
        """
        edges = []
        for i in range(1, len(self)):
            p1 = self.__vertices[i - 1]
            p2 = self.__vertices[i]
            edges.append(Edge(p1, p2))

        return edges

    @property
    def xy(self):
        """
        Returns 2D Array of X&Y-Coordinates
        :return: np.array[np.array[float],np.array[float]]
        """
        x = [p.x for p in self.__vertices]
        y = [p.y for p in self.__vertices]

        return np.array([x, y])

    def __getitem__(self, index) -> Point:
        return self.__vertices[index]

    def __iter__(self):
        return iter(self.__vertices)

    def __str__(self) -> str:
        xy_vertices = [f"{p.x} {p.y}" for p in self.__vertices]

        beginning = "POLYGON (("
        for v in xy_vertices:
            beginning += v + ", "

        result = beginning[:-2] + "))"

        return result

    def __repr__(self) -> str:
        return str(self)

    def to_shapely_geometry(self) -> shapely_geom.Polygon:
        """
        Converts Polygon to Shapely Polygon
        :return: shapely.geometry.Polygon
        """
        coordinates = [p.xy for p in self.__vertices]
        return shapely_geom.Polygon(coordinates)

    @staticmethod
    def from_wkt(wkt_string: str):
        """
        creates polygon from wkt_string
        :param wkt_string:
        :return: Polygon
        """
        n_open_brackets = wkt_string.count("(")
        n_close_brackets = wkt_string.count(")")

        if n_close_brackets + n_close_brackets > 4:
            raise Exception(
                "Sorry, multipolygons and holes are not implemented in this class!")

        else:
            wkt_string = wkt_string.replace("POLYGON ", "")
            wkt_string = wkt_string.replace("POLYGON", "")
            wkt_string = wkt_string.replace("(", "")
            wkt_string = wkt_string.replace(")", "")
            wkt_string = wkt_string.replace(", ", ",")
            coordinates_strings = wkt_string.split(",")

            vertices = []
            for coor_string in coordinates_strings:
                [x_str, y_str] = coor_string.split(" ")
                point = Point([float(x_str), float(y_str)])
                vertices.append(point)
            return Polygon(vertices)

    def bounds(self) -> 'tuple[float,float,float,float]':
        """
        Gets the bounds of the Polygon.
        X-Min, X-Max, Y-Min, Y-Max
        :return: tuple of floats
        """
        polygon_x = [v.x for v in self.vertices]
        polygon_y = [v.y for v in self.vertices]
        xmin = min(polygon_x)
        xmax = max(polygon_x)
        ymin = min(polygon_y)
        ymax = max(polygon_y)

        return xmin, xmax, ymin, ymax

    def area(self) -> float:
        """
        calculates the area of a polygon
        :return:
        """
        # https://www.mathopenref.com/coordpolygonarea2.html
        area = 0
        pj = self[-1]
        for pi in self:
            tri_area = (pj.x + pi.x) * (pj.y - pi.y)
            area += abs(tri_area)
            pj = pi
        return area / 2

    def centroid(self) -> Point:
        """
        calculates the centroid of the polygon
        :return: Point
        """
        p1 = self.__vertices[0]
        p2 = self.__vertices[1]
        x_sum = 0
        y_sum = 0
        area_sum = 0

        for i in range(2, len(self)):
            p3 = self[i]
            triangle_centroid = self.__triangle_centroid(p1, p2, p3)
            triangle_area = self.__triangle_area(p1, p2, p3)
            x_sum += triangle_centroid.x * triangle_area
            y_sum += triangle_centroid.y * triangle_area
            area_sum += triangle_area
            p2 = p3
        x_center = x_sum / area_sum
        y_center = y_sum / area_sum

        return Point([x_center, y_center])

    @staticmethod
    def __triangle_centroid(p1: Point, p2: Point, p3: Point) -> Point:
        """
        calculates the center of a triangle
        :param p1: Point 1
        :param p2: Point 2
        :param p3: Point 3
        :return: Point - Center Point
        """
        xx = (p1.x + p2.x + p3.x) / 3
        yy = (p1.y + p2.y + p3.y) / 3
        return Point([xx, yy])

    @staticmethod
    def __triangle_area(p1: Point, p2: Point, p3: Point) -> float:
        """
        Calculates Area of Triangle
        :param p1: Point 1
        :param p2: Point 2
        :param p3: Point 3
        :return: float, area
        """
        al = p1.distance(p2)
        bl = p2.distance(p3)
        cl = p3.distance(p1)
        s = (al + bl + cl) / 2.0
        area = np.sqrt(s * (s - al) * (s - bl) * (s - cl))
        return area

    def point_in_polygon(self, point: Point) -> bool:
        """
        Tests if a point is inside a Polygon using the Ray casting algorithm
        https://en.wikipedia.org/wiki/Point_in_polygon
        :param point:
        :return: boolean
        """
        # max number,  largest unsigned in 64 bit
        xmax = 2 ** 63
        point_right = Point([xmax, point.y])
        edge_point_right = Edge(point, point_right)
        intersection_count_right = 0

        for poly_segment in self.edges:
            intersects, _ = edge_point_right.intersects(poly_segment)
            if intersects:
                intersection_count_right += 1

        is_in_right = intersection_count_right % 2 != 0

        return is_in_right

    def is_clockwise(self):
        """
        checks if a point is clockwise by summing up the
        crossproduct of the points in the polygon
        https://stackoverflow.com/a/1165943
        if > 0 then ordered clockwise
        if < 0 then counterclockwise
        if == 0 then collinear
        :return:
        """
        res = 0
        for e in self.edges:
            res += np.cross(e.point_2.xy, e.point_1.xy)

        # res > 0 does not handle collinear
        return res > 0

    @staticmethod
    def __build_vertices_list(clip_polygon, subject_polygon) -> Tuple[List[Point], List[str]]:
        """
        helper function for weiler_atherton clipping algorithm
        builds the vertice list and list of point types
        point types are:
            inside: point of subject_polygon inside clip_polygon
            outside: point of subject_polygon outside clip_polygon
            entry: point is an entry point of subject_polygon in clip_polygon
            exit: point is an exit point of subject_polygon from clip_polygon

        :param clip_polygon: Clipping Polygon
        :param subject_polygon: Subject Polygon
        :return: points of polygon with intersections, point types
        """
        polygon_points = []
        vertice_type = []

        for i in range(0, len(subject_polygon) - 1):
            p1 = subject_polygon.vertices[i]
            p2 = subject_polygon.vertices[i + 1]
            p1_inside = clip_polygon.point_in_polygon(p1)
            p2_inside = clip_polygon.point_in_polygon(p2)

            polygon_points.append(p1)

            if p1_inside:
                vertice_type.append("inside")
            else:
                vertice_type.append("outside")

            intersection_points = []
            for e in clip_polygon.edges:
                intersects, intersec_point = Edge(p1, p2).intersects(e)
                if intersects:
                    intersection_points.append(intersec_point)

            for pnt in intersection_points:
                polygon_points.append(pnt)
                if vertice_type[-1] == "inside":
                    vertice_type.append("exit")

                elif vertice_type[-1] == "outside":
                    vertice_type.append("entry")

                elif vertice_type[-1] == "exit":
                    vertice_type.append("entry")

                else:
                    vertice_type.append("exit")

            if (i + 1) == len(subject_polygon) - 1:
                polygon_points.append(p2)

                if p2_inside:
                    vertice_type.append("inside")

                else:
                    vertice_type.append("outside")

        return polygon_points, vertice_type

    def weiler_atherton(self, other):
        """
        Weiler Atherton Clipping Algorithm
        https://en.wikipedia.org/wiki/Weiler%E2%80%93Atherton_clipping_algorithm
        !!!This Implementation can not handle clipping where more than two output polygons would be generated!!!
        :param other: Polygon to be clipped
        :return: Polygon, the clipped polygon
        """
        # https://www.javatpoint.com/weiler-atherton-polygon-clipping

        new_self = self
        new_other = other

        if not other.is_clockwise():
            vertices = other.vertices
            vertices.reverse()
            new_other = Polygon(vertices)

        if not self.is_clockwise():
            vertices = self.vertices
            vertices.reverse()
            new_self = Polygon(vertices)

        subject_vertices, sub_v_types = Polygon.__build_vertices_list(
            new_self, new_other)

        clip_vertices, clip_v_types = Polygon.__build_vertices_list(
            new_other, new_self)

        result = []
        # iterate over the points in subject list
        # if we find an exit point make a 'right turn' and iterate over clipping polygon
        # until clip-poly exits
        for sub_v_type, subj_vertex in zip(sub_v_types, subject_vertices):
            if sub_v_type == "entry":
                result.append(subj_vertex)

            elif sub_v_type == "inside":
                result.append(subj_vertex)

            elif sub_v_type == "exit":
                for idx, clip_start in enumerate(clip_vertices):

                    if clip_start == subj_vertex:
                        for clip_v_type, clip_vertex in zip(clip_v_types[idx:], clip_vertices[idx:]):
                            if clip_v_type == "entry":
                                result.append(clip_vertex)
                            elif clip_v_type == "inside":
                                result.append(clip_vertex)
                            elif clip_v_type == "exit":
                                result.append(clip_vertex)
                                break

        return Polygon(result)


if __name__ == "__main__":
    import test_polygons

    graz_polygon = Polygon.from_wkt(test_polygons.graz_polygon())
    clip_polygon_1 = Polygon.from_wkt(test_polygons.clip_polygon_4())
    res = clip_polygon_1.weiler_atherton(graz_polygon)

    # plt.plot(*res.xy)
    p1 = Point([0, 0])
    p2 = Point([0, 1])
    p3 = Point([1, 1])
    p4 = Point([1, 0])
    polygon = Polygon([p1, p2, p3, p4, p1])

    p5 = Point((1.5, 0))
    p6 = Point((0.5, 0.5))
    p7 = Point((1.5, 2))
    polygon_2 = Polygon([p5, p6, p7, p5])
    polygon.weiler_atherton(polygon_2)
