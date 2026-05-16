"""
=============================================================================
Module: csp_module.py
Description:
    CSP Scheduler / Control Allocation Module for Smart City Traffic &
    Emergency Response AI System.
    Uses Constraint Satisfaction Problem (CSP) solving to assign valid
    signal states (GREEN/YELLOW/RED) to intersections in a control zone
    while respecting safety and conflict constraints.
=============================================================================
"""

SIGNAL_STATES = ["GREEN", "YELLOW", "RED"]

INTERSECTION_GRAPH = {
    "INT_1": ["INT_2", "INT_3"],
    "INT_2": ["INT_1", "INT_4"],
    "INT_3": ["INT_1", "INT_4", "INT_5"],
    "INT_4": ["INT_2", "INT_3", "INT_5"],
    "INT_5": ["INT_3", "INT_4"],
}

ZONE_INTERSECTIONS = {
    "Zone_A": ["INT_1", "INT_2", "INT_3"],
    "Zone_B": ["INT_2", "INT_3", "INT_4"],
    "Zone_C": ["INT_3", "INT_4", "INT_5"],
    "Zone_D": ["INT_1", "INT_2", "INT_4", "INT_5"],
    "Zone_E": ["INT_1", "INT_2", "INT_3", "INT_4", "INT_5"],
}


def _get_next_intersection(intersections: list, assignment: dict) -> str:
    """Return the next unassigned intersection."""
    for intersection in intersections:
        if intersection not in assignment:
            return intersection
    return ""


def _get_states_to_try(emergency_mode: bool, assignment: dict) -> list:
    """Return the state order for the next decision."""
    if emergency_mode and not assignment:
        return ["GREEN", "YELLOW", "RED"]
    return list(SIGNAL_STATES)


def _count_signals(assignment: dict) -> dict:
    """Count how many intersections received each signal state."""
    return {
        "GREEN": sum(1 for state in assignment.values() if state == "GREEN"),
        "YELLOW": sum(1 for state in assignment.values() if state == "YELLOW"),
        "RED": sum(1 for state in assignment.values() if state == "RED"),
    }


def is_consistent(intersection: str, state: str, assignment: dict) -> bool:
    """
    -------------------------------------------------------------------------
    Function: is_consistent
    Description:
        Checks whether assigning 'state' to 'intersection' is consistent
        with the current partial assignment.
        Constraint: Two adjacent intersections cannot both be GREEN.
    -------------------------------------------------------------------------
    """
    if state != "GREEN":
        return True
    for neighbor in INTERSECTION_GRAPH.get(intersection, []):
        if assignment.get(neighbor) == "GREEN":
            return False
    return True


def backtrack(intersections: list, assignment: dict, emergency_mode: bool):
    """
    -------------------------------------------------------------------------
    Function: backtrack
    Description:
        Recursive backtracking algorithm for CSP.
        Returns a complete valid assignment or None if unsolvable.
    -------------------------------------------------------------------------
    """
    if len(assignment) == len(intersections):
        return assignment

    current = _get_next_intersection(intersections, assignment)
    for state in _get_states_to_try(emergency_mode, assignment):
        if not is_consistent(current, state, assignment):
            continue
        assignment[current] = state
        result = backtrack(intersections, assignment, emergency_mode)
        if result is not None:
            return result
        del assignment[current]

    return None


def solve_csp(control_zone: str, emergency_mode: bool = False, approved_action: str = None) -> dict:
    """
    -------------------------------------------------------------------------
    Function: solve_csp
    Description:
        Main entry point for the CSP module.
        Identifies intersections in the given control zone, runs backtracking
        CSP solver, and returns the signal assignment plan with explanation.
    -------------------------------------------------------------------------
    """
    intersections = ZONE_INTERSECTIONS.get(control_zone, [])
    if not intersections:
        return {
            "success": False,
            "control_zone": control_zone,
            "assignment": {},
            "explanation": f"No intersections found for zone '{control_zone}'.",
        }

    assignment = backtrack(intersections, {}, emergency_mode)
    if assignment is None:
        return {
            "success": False,
            "control_zone": control_zone,
            "assignment": {},
            "explanation": "CSP solver could not find a valid signal assignment.",
        }

    signal_counts = _count_signals(assignment)
    explanation_parts = [
        f"Signal plan computed for {len(intersections)} intersections in {control_zone}.",
        (
            "Emergency mode active: corridor cleared with GREEN priority."
            if emergency_mode else None
        ),
        (
            f"Active GREEN signals: {signal_counts['GREEN']} | "
            f"YELLOW: {signal_counts['YELLOW']} | RED: {signal_counts['RED']}"
        ),
        f"Action applied: {approved_action}" if approved_action else None,
    ]

    return {
        "success": True,
        "control_zone": control_zone,
        "assignment": assignment,
        "green_corridor": [node for node, state in assignment.items() if state == "GREEN"],
        "explanation": " | ".join(part for part in explanation_parts if part),
    }
