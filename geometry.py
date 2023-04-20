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

    def point_1(self) -> Point:
        return self.vertices[0]

    def point_2(self) -> Point:
        return self.vertices[1]

    def length(self) -> float:
        self.point_1.distance(self.point_2)

    def __str__(self) -> str:
        (x1, y1) = self.point_1.xy
        (x2, y2) = self.point_2.xy
        return f"LINESTRING ({x1} {y1}, {x2} {y2})"


class Polygon():
    def __init__(self, vertices) -> None:
        self.__vertices = vertices

    def __len__(self):
        return len(self.__vertices)

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
            raise Exception("Sorry, multipolygons and holes are not implementet in this class!")
        
        else:
            wkt_string = wkt_string.replace("POLYGON ","")
            wkt_string = wkt_string.replace("POLYGON","")
            wkt_string = wkt_string.replace("(","")
            wkt_string = wkt_string.replace(")","")
            wkt_string = wkt_string.replace(", ", ",")
            coordinates_strings = wkt_string.split(",")

            vertices = []
            for coor_string in coordinates_strings:
                [x_str,y_str] = coor_string.split(" ")
                point = Point([float(x_str),float(y_str)])
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