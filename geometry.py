import numpy as np


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

    @property
    def xy(self):
        return self.x, self.y

    def distance(self, other) -> float:
        a = self.x - other.x
        b = self.y - other.y
        return np.sqrt(a**2 + b**2)

    def __str__(self) -> str:
        return f"POINT ({self.x} {self.y})"

    def __eq__(self, other) -> bool:
        return (self.x == other.x) & (self.y == other.y)


class Line():
    def __init__(self, point1: Point, point2: Point) -> None:
        self.vertices = (point1, point2)

    @property
    def point_1(self) -> Point:
        return self.vertices[0]

    @property
    def point_2(self) -> Point:
        return self.vertices[1]

    def length(self) -> float:
        return self.point_1.distance(self.point_2)

    def __str__(self) -> str:
        (x1, y1) = self.point_1.xy
        (x2, y2) = self.point_2.xy
        return f"LINESTRING ({x1} {y1}, {x2} {y2})"

    def intersects(self, other):
        # cross product method to check if line intersects
        # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
        # runs from p -> p+r and second line q to q+s

        # initial two startingpoints of the line
        p = np.array([self.point_1.x, self.point_1.y])
        q = np.array([other.point_1.x, other.point_1.y])

        # the points p1 and q1 can be expressed as p + r and q + s
        p_plus_r = np.array([self.point_2.x, self.point_2.y])
        q_plus_s = np.array([other.point_2.x, other.point_2.y])

        # sooo
        r = p_plus_r - p
        s = q_plus_s - q

        # any point on first line is p + t * r
        # any point on second line is q + u * s
        # intersects if we can find t and u such that: p + t*r = q+u*s
        # t = (q - p) × s / (r × s)
        # u = (q - p) × r / (r × s)

        r_cross_s = np.cross(r, s)

        q_min_p_cross_s = np.cross((q - p), s)
        q_min_p_cross_r = np.cross((q - p), r)

        t = q_min_p_cross_s / r_cross_s
        if np.isnan(t):
            return (False, None)

        u = q_min_p_cross_r / r_cross_s

        if np.isnan(u):
            return (False, None)

        if r_cross_s == 0 and q_min_p_cross_r == 0:

            """
            check the points if intersection
            took me a while to figure this out but it gets easier

            comparing the signs of the differences between the x and y
            coordinates of each pair of endpoints. 
            If all differences have the same sign, it means that the line segments overlap
            and intersect.    
            if you draw it https://www.geogebra.org/calculator/nucy4npt
            """
            # collinear
            t_zero = (q-p) @ r / (r @ r)
            t_one = (q + s - p) @ r / (r @ r) + t_zero + s @ r / r @ r

            p_1 = Point(p)
            p_2 = Point(q)  # rename for ease
            q_1 = Point(q_plus_s)
            q_2 = Point(p_plus_r)

            # are points the same
            for pi in [p_1, p_2]:
                for pj in [q_1, q_2]:
                    if pi == pj:
                        return (True, pi)

            else:
                # lines do not touch but do they overlap?
                # check if x or y values overlap somewhere
                x_vals = [q_1.x, q_2.x, p_1.x, p_2.x]
                y_vals = [q_1.y, q_2.y, p_1.y, p_2.y]
                overlap = (min(x_vals) < max(x_vals)) and (
                    min(y_vals) < max(y_vals))
                if overlap:
                    return (True, q + u*s)
                else:
                    return (False, None)

        elif r_cross_s == 0 and q_min_p_cross_r != 0:
            # parallel and non-intersecting
            return (False, None)

        elif (r_cross_s != 0) and (0 <= t <= 1) and 0 <= u <= 1:
            # lines intersect at point q + u*s
            return (True, q + u * s)

        else:
            # non-intersecting and non-parallel
            return (False, None)


class Polygon():
    def __init__(self, vertices) -> None:
        self.__vertices = vertices

    def __len__(self):
        return len(self.__vertices)

    @property
    def veritces(self):
        return self.__vertices

    def __getitem__(self, index):
        return self.__vertices[index]

    def __iter__(self):
        return iter(self.__vertices)

    def __str__(self):
        xy_vertices = [f"{p.x} {p.y}" for p in self.__vertices]

        beginning = "POLYGON (("
        for v in xy_vertices:
            beginning += v + ", "

        result = beginning[:-2] + "))"
        return result

    @staticmethod
    def from_wkt(wkt_string):
        n_open_brackets = wkt_string.count("(")
        n_close_brackets = wkt_string.count(")")

        if n_close_brackets + n_close_brackets > 4:
            raise Exception(
                "Sorry, multipolygons and holes are not implementet in this class!")

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

    def area(self):
        # https://www.mathopenref.com/coordpolygonarea2.html
        area = 0
        pj = self[-1]
        for pi in self:
            res = (pj.x + pi.x) * (pj.y - pi.y)
            area += abs(res)
            pj = pi
        return area/2

    def centroid(self):
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
    def __triangle_centroid(p1, p2, p3):
        xx = (p1.x + p2.x + p3.x) / 3
        yy = (p1.y + p2.y + p3.y) / 3
        return Point([xx, yy])

    @staticmethod
    def __triangle_area(p1, p2, p3):

        al = p1.distance(p2)
        bl = p2.distance(p3)
        cl = p3.distance(p1)
        s = (al + bl + cl) / 2.0
        area = np.sqrt(s * (s - al) * (s - bl) * (s - cl))
        return area

    def __weiler_atherton(self, other):
        polygon_a_vertices = self.veritces
        polygon_b_vertices = other.vertices
        """
        To do
        """

        pass


if __name__ == "__main__":
    p1 = Point([0, 2])
    p2 = Point([2, 2])
    p3 = Point([2, 0])
    p4 = Point([0, 0])

    poly = Polygon([p1, p2, p3, p4])
    print(poly.centroid())
    print(poly.area())
    print(poly)
    wkt = "POLYGON ((0 2, 2 2, 2 0, 0 0))"
    poly = Polygon.from_wkt(wkt)
    print(poly)

    p1 = Point([-2, 0])
    p2 = Point([3, 2])
    q1 = Point([0, 3])
    q2 = Point([4, 0])

    line_a = Line(p1, p2)
    line_b = Line(q1, q2)

    print(line_a.intersects(line_b))

    p1 = Point([0, 0])
    p2 = Point([1, 1])
    q1 = Point([0, 1])
    q2 = Point([1, 0])
    line_a = Line(p1, p2)
    line_b = Line(q1, q2)
    print(line_a.intersects(line_b))

    p1 = Point([0, 0])
    p2 = Point([1, 1])
    q1 = Point([0, 2])
    q2 = Point([1, 3])
    line_a = Line(p1, p2)
    line_b = Line(q1, q2)
    print(line_a.intersects(line_b))

    p1 = Point([0, 0])
    p2 = Point([0, 1])
    q1 = Point([1, 0])
    q2 = Point([1, 1])
    line_a = Line(p1, p2)
    line_b = Line(q1, q2)
    print(line_a.intersects(line_b))

    p1 = Point([0, 0])
    p2 = Point([1, 1])
    q1 = Point([0, 0.5])
    q2 = Point([1, 0.5])
    line_a = Line(p1, p2)
    line_b = Line(q1, q2)
    print(line_a.intersects(line_b))

    p1 = Point([0, 0])
    p2 = Point([2, 2])
    q1 = Point([1, 1])
    q2 = Point([3, 3])
    line_a = Line(p1, p2)
    line_b = Line(q1, q2)
    print(line_a.intersects(line_b))
