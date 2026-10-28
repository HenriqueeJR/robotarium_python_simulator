from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon


def construir_grafo_topologico(corredores, rotatorias):
    """Build the directed corridor/roundabout graph used by route planning."""
    polygons = [Polygon(item["v"]) for item in corredores]
    node_types = ["C"] * len(corredores)
    original_indices = list(range(1, len(corredores) + 1))
    names = [f"C{i + 1}" for i in range(len(corredores))]
    centers = [tuple(polygon.centroid.coords[0]) for polygon in polygons]
    directions = [item["dir"] for item in corredores]

    angles = np.linspace(0.0, 2.0 * np.pi, 32)
    for index, roundabout in enumerate(rotatorias):
        cx, cy, outer_radius = roundabout[:3]
        polygons.append(Polygon(np.column_stack((
            cx + outer_radius * np.cos(angles),
            cy + outer_radius * np.sin(angles),
        ))))
        node_types.append("R")
        original_indices.append(index + 1)
        names.append(f"R{index + 1}")
        centers.append((cx, cy))
        directions.append(None)

    adjacency = [[] for _ in polygons]
    for source, source_polygon in enumerate(polygons):
        for target, target_polygon in enumerate(polygons):
            if source == target:
                continue
            intersection = source_polygon.intersection(target_polygon)
            if intersection.area <= 1e-4:
                continue
            point = intersection.centroid
            point_xy = (point.x, point.y)
            valid = True
            if node_types[source] == "C":
                direction = directions[source]
                dot = ((point_xy[0] - centers[source][0]) * direction[0]
                       + (point_xy[1] - centers[source][1]) * direction[1])
                valid = dot >= -1e-3
            if valid and node_types[target] == "C":
                direction = directions[target]
                dot = ((centers[target][0] - point_xy[0]) * direction[0]
                       + (centers[target][1] - point_xy[1]) * direction[1])
                valid = dot >= -1e-3
            if valid:
                adjacency[source].append((target, math.dist(centers[source], centers[target])))

    nodes = [
        {"name": names[index], "type": node_types[index],
         "original_index": original_indices[index], "center": centers[index]}
        for index in range(len(polygons))
    ]
    return adjacency, polygons, centers, nodes
