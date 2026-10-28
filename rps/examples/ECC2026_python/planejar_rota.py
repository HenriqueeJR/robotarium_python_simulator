from __future__ import annotations

import heapq

import numpy as np
from shapely.geometry import Point

from .construir_grafo_topologico import construir_grafo_topologico


def planejar_rota(corredores, rotatorias, pt_start, pt_target):
    adjacency, polygons, centers, nodes = construir_grafo_topologico(
        corredores, rotatorias
    )
    start_node = next(
        (i for i, polygon in enumerate(polygons)
         if polygon.covers(Point(*pt_start))), None
    )
    target_node = next(
        (i for i, polygon in enumerate(polygons)
         if polygon.covers(Point(*pt_target))), None
    )
    if start_node is None or target_node is None:
        raise ValueError("Start and target must lie inside the road polygons.")

    distances = [float("inf")] * len(nodes)
    previous = [None] * len(nodes)
    distances[start_node] = 0.0
    queue = [(0.0, start_node)]
    while queue:
        current_distance, source = heapq.heappop(queue)
        if current_distance != distances[source]:
            continue
        if source == target_node:
            break
        for target, weight in adjacency[source]:
            candidate = current_distance + weight
            if candidate < distances[target]:
                distances[target] = candidate
                previous[target] = source
                heapq.heappush(queue, (candidate, target))

    if not np.isfinite(distances[target_node]):
        raise ValueError("No feasible directed road route exists.")
    path_nodes = []
    node = target_node
    while node is not None:
        path_nodes.append(node)
        if node == start_node:
            break
        node = previous[node]
    path_nodes.reverse()
    if not path_nodes or path_nodes[0] != start_node:
        raise ValueError("No feasible directed road route exists.")

    path_names = [nodes[index]["name"] for index in path_nodes]
    corridor_nodes = [nodes[index]["original_index"] for index in path_nodes
                      if nodes[index]["type"] == "C"]
    roundabout_nodes = [nodes[index]["original_index"] for index in path_nodes
                        if nodes[index]["type"] == "R"]
    selected_corridors = [corredores[index - 1] for index in corridor_nodes]
    selected_roundabouts = np.asarray(
        [rotatorias[index - 1] for index in roundabout_nodes], dtype=float
    ).reshape(-1, 5)

    connections = []
    for index in range(len(path_nodes) - 2):
        first, middle, last = (nodes[path_nodes[index + offset]]
                               for offset in range(3))
        if (first["type"], middle["type"], last["type"]) == ("C", "R", "C"):
            connections.append((
                corridor_nodes.index(first["original_index"]) + 1,
                corridor_nodes.index(last["original_index"]) + 1,
                roundabout_nodes.index(middle["original_index"]) + 1,
            ))

    return (
        selected_corridors,
        selected_roundabouts,
        np.asarray(connections, dtype=int).reshape(-1, 3),
        path_names,
        adjacency,
        np.asarray(centers, dtype=float),
        np.asarray(path_nodes, dtype=int) + 1,
    )
