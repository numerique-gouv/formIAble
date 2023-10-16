from typing import List, Union
import numpy as np
import scipy


def segment_intersection(vertices_segment1: np.ndarray,
                         vertices_segment2: np.ndarray) -> Union[None, np.ndarray]:
    assert vertices_segment1.shape == (2, 2)
    assert vertices_segment2.shape == (2, 2)

    det: float = np.linalg.det([vertices_segment1[0, :] - vertices_segment1[1, :],
                                vertices_segment2[1, :] - vertices_segment2[0, :]])
    if det == 0:
        return None
    t: float = np.linalg.det([vertices_segment2[1, :] - vertices_segment1[1, :],
                              vertices_segment2[1, :] - vertices_segment2[0, :]]) / det
    t_prime: float = np.linalg.det([vertices_segment1[0, :] - vertices_segment1[1, :],
                                    vertices_segment2[1, :] - vertices_segment1[1, :]]) / det
    if 0 <= t <= 1 and 0 <= t_prime <= 1:
        return vertices_segment1[0, :] * t + vertices_segment1[1, :] * (1 - t)
    else:
        return None


def find_intersecting_points_convex_polygons(vertices_polygon1: Union[np.ndarray, List],
                                             vertices_polygon2: Union[np.ndarray, List],
                                             epsilon: float = 1e-6) -> List[np.ndarray]:
    """Find the intersecting points of two convex polygons

    A point is considered an intersecting points if including it in the other polygon
    does not increase the volume of its convex hull by more than epsilon (numerical error)
    or it is an intersection point of two non-parallel polygon segments
    """
    vertices_polygon1_ = [np.array(vertex) for vertex in vertices_polygon1]
    vertices_polygon2_ = [np.array(vertex) for vertex in vertices_polygon2]
    vertices_polygon1 = vertices_polygon1_.copy()
    vertices_polygon2 = vertices_polygon2_.copy()

    intersecting_points: List[np.ndarray] = []

    # Add find intersecting vertices
    for polygon_to_look_in, other_polygon in zip([vertices_polygon1_, vertices_polygon2_],
                                                 [vertices_polygon2, vertices_polygon1]):
        area_polygon_to_look_in: float = scipy.spatial.ConvexHull(polygon_to_look_in).volume
        for vertex in other_polygon:
            polygon_to_look_in.append(vertex)
            new_area: float = scipy.spatial.ConvexHull(polygon_to_look_in).volume
            polygon_to_look_in.pop()
            if new_area <= area_polygon_to_look_in + epsilon:
                intersecting_points.append(vertex)

    # convert to array to enable slicing with a list of indexes
    vertices_polygon1_ = np.array(vertices_polygon1)
    vertices_polygon2_ = np.array(vertices_polygon2)

    # loop over the two polygons to find remaining intersections
    previous_index_polygon1: int = -1
    for index_polygon1 in range(len(vertices_polygon1_)):
        previous_index_polygon2: int = -1
        for index_polygon2 in range(len(vertices_polygon2_)):
            intersection: Union[None, np.ndarray] = segment_intersection(
                vertices_polygon1_[[previous_index_polygon1, index_polygon1], :],
                vertices_polygon2_[[previous_index_polygon2, index_polygon2], :]
            )
            if intersection is not None:
                intersecting_points.append(intersection)
            previous_index_polygon2 = index_polygon2
        previous_index_polygon1 = index_polygon1
    return intersecting_points


def area_intersection_convex_polygons(vertices_polygon1: Union[np.ndarray, List],
                                      vertices_polygon2: Union[np.ndarray, List],
                                      epsilon: float = 1e-6) -> float:
    """Compute the area of the intersection of two convex polygons"""

    intersecting_points: List[np.ndarray] = find_intersecting_points_convex_polygons(vertices_polygon1,
                                                                                     vertices_polygon2,
                                                                                     epsilon)
    return scipy.spatial.ConvexHull(intersecting_points).volume
