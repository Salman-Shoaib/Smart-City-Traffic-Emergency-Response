"""
=============================================================================
Module: search_module.py
Description:
    Search & Navigation Module for Smart City Traffic & Emergency Response
    AI System.
    Implements BFS (unweighted), UCS (weighted), and A* (heuristic) search
    algorithms on the city road network graph. Returns the optimal route
    from source to destination based on request type.
=============================================================================
"""

import heapq
from collections import deque


CITY_GRAPH_UNWEIGHTED = {
    "Central_Junction": ["North_Gate", "South_Gate", "East_Avenue", "West_Boulevard", "Market_Square"],
    "North_Gate": ["Central_Junction", "City_Hospital", "Fire_Station"],
    "South_Gate": ["Central_Junction", "Police_HQ", "Industrial_Zone"],
    "East_Avenue": ["Central_Junction", "Industrial_Zone", "Market_Square"],
    "West_Boulevard": ["Central_Junction", "Market_Square", "Fire_Station"],
    "Market_Square": ["Central_Junction", "East_Avenue", "West_Boulevard"],
    "City_Hospital": ["North_Gate", "Fire_Station"],
    "Fire_Station": ["North_Gate", "West_Boulevard", "City_Hospital"],
    "Police_HQ": ["South_Gate"],
    "Industrial_Zone": ["South_Gate", "East_Avenue"],
}

CITY_GRAPH_WEIGHTED = {
    "Central_Junction": [("North_Gate", 3), ("South_Gate", 3), ("East_Avenue", 4), ("West_Boulevard", 4), ("Market_Square", 2)],
    "North_Gate": [("Central_Junction", 3), ("City_Hospital", 6), ("Fire_Station", 4)],
    "South_Gate": [("Central_Junction", 3), ("Police_HQ", 3), ("Industrial_Zone", 5)],
    "East_Avenue": [("Central_Junction", 4), ("Industrial_Zone", 5), ("Market_Square", 3)],
    "West_Boulevard": [("Central_Junction", 4), ("Market_Square", 3), ("Fire_Station", 5)],
    "Market_Square": [("Central_Junction", 2), ("East_Avenue", 3), ("West_Boulevard", 3)],
    "City_Hospital": [("North_Gate", 6), ("Fire_Station", 2)],
    "Fire_Station": [("North_Gate", 4), ("West_Boulevard", 5), ("City_Hospital", 2)],
    "Police_HQ": [("South_Gate", 3)],
    "Industrial_Zone": [("South_Gate", 5), ("East_Avenue", 5)],
}

HEURISTIC_TO_HOSPITAL = {
    "Central_Junction": 5,
    "North_Gate": 2,
    "South_Gate": 8,
    "East_Avenue": 6,
    "West_Boulevard": 5,
    "Market_Square": 6,
    "City_Hospital": 0,
    "Fire_Station": 1,
    "Police_HQ": 9,
    "Industrial_Zone": 9,
}
HEURISTIC_GENERIC = {node: 3 for node in CITY_GRAPH_WEIGHTED}

ROUTE_REASON_BY_ALGORITHM = {
    "A*": "A* chosen: emergency requires fastest route.",
    "BFS": "BFS chosen: simple unweighted route for civilian.",
    "UCS": "UCS chosen: optimal weighted route.",
}


def _failure_result(algorithm: str, message: str) -> dict:
    """Build a standard failure response."""
    return {
        "success": False,
        "algorithm": algorithm,
        "path": [],
        "cost": 0,
        "explanation": message,
    }


def _success_result(algorithm: str, path: list, cost: int, message: str) -> dict:
    """Build a standard success response."""
    return {
        "success": True,
        "algorithm": algorithm,
        "path": path,
        "cost": cost,
        "explanation": message,
    }


def _validate_graph_node(graph: dict, node_name: str, node_value: str, algorithm: str):
    """Return a failure result when a node is not in the graph."""
    if node_value not in graph:
        return _failure_result(algorithm, f"{node_name} '{node_value}' not in graph.")
    return None


def get_heuristic(node: str, destination: str) -> int:
    """
    -------------------------------------------------------------------------
    Function: get_heuristic
    Description:
        Returns the heuristic estimate from a node to the destination.
    -------------------------------------------------------------------------
    """
    if destination == "City_Hospital":
        return HEURISTIC_TO_HOSPITAL.get(node, 5)
    return HEURISTIC_GENERIC.get(node, 3)


def bfs(source: str, destination: str) -> dict:
    """
    -------------------------------------------------------------------------
    Function: bfs
    Description:
        Breadth-First Search on the unweighted city graph.
    -------------------------------------------------------------------------
    """
    source_error = _validate_graph_node(CITY_GRAPH_UNWEIGHTED, "Source", source, "BFS")
    if source_error:
        return source_error
    destination_error = _validate_graph_node(CITY_GRAPH_UNWEIGHTED, "Destination", destination, "BFS")
    if destination_error:
        return destination_error

    queue = deque([[source]])
    visited = {source}

    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == destination:
            cost = len(path) - 1
            return _success_result("BFS", path, cost, f"BFS found path with {cost} hops.")

        for neighbor in CITY_GRAPH_UNWEIGHTED.get(current, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(path + [neighbor])

    return _failure_result("BFS", f"No path found from '{source}' to '{destination}'.")


def ucs(source: str, destination: str) -> dict:
    """
    -------------------------------------------------------------------------
    Function: ucs
    Description:
        Uniform Cost Search on the weighted city graph.
    -------------------------------------------------------------------------
    """
    source_error = _validate_graph_node(CITY_GRAPH_WEIGHTED, "Source", source, "UCS")
    if source_error:
        return source_error
    destination_error = _validate_graph_node(CITY_GRAPH_WEIGHTED, "Destination", destination, "UCS")
    if destination_error:
        return destination_error

    heap = [(0, source, [source])]
    visited = set()

    while heap:
        cost, current, path = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)

        if current == destination:
            return _success_result("UCS", path, cost, f"UCS found optimal path. Total cost: {cost} units.")

        for neighbor, edge_cost in CITY_GRAPH_WEIGHTED.get(current, []):
            if neighbor not in visited:
                heapq.heappush(heap, (cost + edge_cost, neighbor, path + [neighbor]))

    return _failure_result("UCS", f"No path found from '{source}' to '{destination}'.")


def astar(source: str, destination: str) -> dict:
    """
    -------------------------------------------------------------------------
    Function: astar
    Description:
        A* Search using f(n) = g(n) + h(n).
    -------------------------------------------------------------------------
    """
    source_error = _validate_graph_node(CITY_GRAPH_WEIGHTED, "Source", source, "A*")
    if source_error:
        return source_error
    destination_error = _validate_graph_node(CITY_GRAPH_WEIGHTED, "Destination", destination, "A*")
    if destination_error:
        return destination_error

    heap = [(get_heuristic(source, destination), 0, source, [source])]
    best_cost_by_node = {}

    while heap:
        f_cost, g_cost, current, path = heapq.heappop(heap)
        if current in best_cost_by_node and best_cost_by_node[current] <= g_cost:
            continue
        best_cost_by_node[current] = g_cost

        if current == destination:
            return _success_result("A*", path, g_cost, f"A* found optimal path. Travel cost: {g_cost} units.")

        for neighbor, edge_cost in CITY_GRAPH_WEIGHTED.get(current, []):
            new_g_cost = g_cost + edge_cost
            new_f_cost = new_g_cost + get_heuristic(neighbor, destination)
            heapq.heappush(heap, (new_f_cost, new_g_cost, neighbor, path + [neighbor]))

    return _failure_result("A*", f"No path found from '{source}' to '{destination}'.")


def _select_algorithm(request: dict):
    """Return the search function and reason text for the request."""
    request_category = request.get("request_category", "Route_Request")
    vehicle_type = request.get("vehicle_type", "civilian")

    if request_category in ["Emergency_Response_Request", "Integrated_City_Service_Request"]:
        return astar, ROUTE_REASON_BY_ALGORITHM["A*"]
    if request_category == "Route_Request" and vehicle_type == "civilian":
        return bfs, ROUTE_REASON_BY_ALGORITHM["BFS"]
    return ucs, ROUTE_REASON_BY_ALGORITHM["UCS"]


def find_route(source: str, destination: str, request: dict) -> dict:
    """
    -------------------------------------------------------------------------
    Function: find_route
    Description:
        Selects the appropriate search algorithm based on request type.
    -------------------------------------------------------------------------
    """
    search_function, algorithm_reason = _select_algorithm(request)
    result = search_function(source, destination)
    result["algorithm_reason"] = algorithm_reason
    return result
