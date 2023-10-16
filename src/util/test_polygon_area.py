import numpy as np
import src.util.polygon_area
import logging
logging.basicConfig(level=logging.DEBUG)


val = src.util.polygon_area.area_intersection_convex_polygons(vertices_polygon1=np.array([[0, 0],
                                                                                          [2, 0],
                                                                                          [2, 1],
                                                                                          [0, 1]]),
                                                              vertices_polygon2=np.array([[0, 0],
                                                                                          [1, 0],
                                                                                          [1, 2],
                                                                                          [0, 2]]))

logging.debug(val)

logging.debug(
    src.util.polygon_area.segment_intersection(vertices_segment1=np.array([[2, 0], [2, 1]]),
                                               vertices_segment2=np.array([[0, 0], [1, 0]]))
)
