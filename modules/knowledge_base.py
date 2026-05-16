"""
=============================================================================
Module: knowledge_base.py
Description:
    Logic / Knowledge Base Module for Smart City Traffic & Emergency
    Response AI System.
    Implements rule-based reasoning using predicate logic to validate
    traffic-control actions, check policy compliance, and determine
    authorization for emergency operations.
=============================================================================
"""

EMERGENCY_VEHICLES = {"ambulance", "fire_truck", "police"}
CIVILIAN_VEHICLES = {"civilian"}
HOSPITAL_DESTINATIONS = {"City_Hospital"}
SIGNAL_ZONES = {"Zone_A", "Zone_B", "Zone_C", "Zone_D", "Zone_E"}
VALID_ACTIONS = {"SignalOverride", "EmergencyRoute", "LaneControl", "CorridorAccess"}


def is_emergency_vehicle(vehicle_type: str) -> bool:
    """Predicate: returns True if vehicle_type is an emergency vehicle."""
    return vehicle_type.lower() in EMERGENCY_VEHICLES


def is_civilian_vehicle(vehicle_type: str) -> bool:
    """Predicate: returns True if vehicle_type is a civilian vehicle."""
    return vehicle_type.lower() in CIVILIAN_VEHICLES


def is_high_severity(severity: str) -> bool:
    """Predicate: returns True if severity is high or critical."""
    return severity.lower() in {"high", "critical"}


def is_time_sensitive(time_sensitivity: str) -> bool:
    """Predicate: returns True if time_sensitivity is high."""
    return time_sensitivity.lower() == "high"


def is_hospital_destination(destination: str) -> bool:
    """Predicate: returns True if destination is a hospital."""
    return destination in HOSPITAL_DESTINATIONS


def in_signal_zone(control_zone: str) -> bool:
    """Predicate: returns True if control_zone is registered."""
    return control_zone in SIGNAL_ZONES


def determine_priority(vehicle_type: str, severity: str, time_sensitivity: str) -> str:
    """
    -------------------------------------------------------------------------
    Function: determine_priority
    Description:
        Applies priority inference rules from the knowledge base.
    -------------------------------------------------------------------------
    """
    if not is_emergency_vehicle(vehicle_type):
        return "Normal"
    if is_high_severity(severity):
        return "Critical"
    return "High"


def check_signal_override_authorization(vehicle_type: str, control_zone: str) -> tuple:
    """
    -------------------------------------------------------------------------
    Function: check_signal_override_authorization
    Description:
        Applies authorization rules for signal override.
    -------------------------------------------------------------------------
    """
    if not in_signal_zone(control_zone):
        return False, f"Zone '{control_zone}' is not a registered signal zone."
    if is_emergency_vehicle(vehicle_type):
        return True, f"Emergency vehicle authorized for SignalOverride in {control_zone}."
    return False, "Civilian vehicles are NOT authorized for signal override."


def check_emergency_corridor(vehicle_type: str, destination: str) -> tuple:
    """
    -------------------------------------------------------------------------
    Function: check_emergency_corridor
    Description:
        Applies authorization rules for emergency corridor access.
    -------------------------------------------------------------------------
    """
    if not is_emergency_vehicle(vehicle_type):
        return False, "No emergency corridor. Civilian vehicle standard routing applies."
    if is_hospital_destination(destination):
        return True, f"Emergency corridor authorized: route to {destination}."
    return True, "Emergency corridor authorized for emergency vehicle."


def validate_action(vehicle_type: str, action: str, control_zone: str, destination: str) -> tuple:
    """
    -------------------------------------------------------------------------
    Function: validate_action
    Description:
        Checks whether the requested action is allowed for the vehicle.
    -------------------------------------------------------------------------
    """
    if action not in VALID_ACTIONS:
        return False, f"Unknown action '{action}'. Valid: {VALID_ACTIONS}"
    if action == "SignalOverride":
        return check_signal_override_authorization(vehicle_type, control_zone)
    if action in {"EmergencyRoute", "CorridorAccess"}:
        return check_emergency_corridor(vehicle_type, destination)
    if action == "LaneControl":
        if is_emergency_vehicle(vehicle_type):
            return True, "Emergency vehicle authorized for LaneControl."
        return False, "Civilian vehicle not authorized for LaneControl."
    return False, "Action not recognized."


def _make_policy_result() -> dict:
    """Create the default knowledge-base response shape."""
    return {
        "approved": False,
        "policy_priority": None,
        "signal_override_allowed": False,
        "emergency_corridor": False,
        "rules_fired": [],
        "explanation": [],
    }


def _record_rule(result: dict, label: str, message: str):
    """Append a fired rule message to the result."""
    result["rules_fired"].append(f"{label} -> {message}")


def _approve_route_request(result: dict):
    result["approved"] = True
    result["explanation"].append("Route_Request: Always approved per policy.")


def _approve_policy_check(result: dict, corridor_ok: bool, signal_ok: bool):
    result["approved"] = corridor_ok or signal_ok
    if result["approved"]:
        result["explanation"].append("Policy_Check: Authorization confirmed.")
    else:
        result["explanation"].append("Policy_Check: No authorization found. Rejected.")


def _approve_control_allocation(result: dict, signal_ok: bool):
    result["approved"] = signal_ok
    if signal_ok:
        result["explanation"].append("Control_Allocation: SignalOverride authorized.")
    else:
        result["explanation"].append("Control_Allocation: Not authorized. Rejected.")


def _approve_emergency_response(result: dict, vehicle_type: str, kb_priority: str, corridor_ok: bool):
    result["approved"] = is_emergency_vehicle(vehicle_type) and corridor_ok
    if result["approved"]:
        result["explanation"].append(
            f"Emergency_Response: Priority({kb_priority}) + Corridor authorized."
        )
    else:
        result["explanation"].append("Emergency_Response: Authorization failed. Rejected.")


def _approve_integrated_service(
    result: dict,
    vehicle_type: str,
    kb_priority: str,
    corridor_ok: bool,
    signal_ok: bool,
):
    if kb_priority == "Critical" and corridor_ok and signal_ok:
        result["approved"] = True
        result["explanation"].append(
            "Integrated_City_Service: Critical priority + corridor + signal override approved."
        )
        return
    if is_emergency_vehicle(vehicle_type) and corridor_ok:
        result["approved"] = True
        result["explanation"].append(
            "Integrated_City_Service: Emergency vehicle with corridor approved."
        )
        return
    result["approved"] = False
    result["explanation"].append("Integrated_City_Service: Insufficient authorization.")


def validate_request_policy(request: dict) -> dict:
    """
    -------------------------------------------------------------------------
    Function: validate_request_policy
    Description:
        Main entry point for the Knowledge Base module.
        Applies all relevant rules based on request category and vehicle type.
        Returns a policy result dict with approval status and explanation.
    -------------------------------------------------------------------------
    """
    vehicle_type = request["vehicle_type"]
    destination = request["destination"]
    control_zone = request["control_zone"]
    severity = request["incident_severity"]
    time_sensitivity = request["time_sensitivity"]
    request_category = request["request_category"]

    result = _make_policy_result()
    kb_priority = determine_priority(vehicle_type, severity, time_sensitivity)
    corridor_ok, corridor_msg = check_emergency_corridor(vehicle_type, destination)
    signal_ok, signal_msg = check_signal_override_authorization(vehicle_type, control_zone)

    result["policy_priority"] = kb_priority
    result["emergency_corridor"] = corridor_ok
    result["signal_override_allowed"] = signal_ok

    _record_rule(result, "Priority Rule", f"{vehicle_type} => Priority({kb_priority})")
    _record_rule(result, "Corridor Rule", corridor_msg)
    _record_rule(result, "Signal Override Rule", signal_msg)

    if request_category == "Route_Request":
        _approve_route_request(result)
    elif request_category == "Policy_Check":
        _approve_policy_check(result, corridor_ok, signal_ok)
    elif request_category == "Control_Allocation_Request":
        _approve_control_allocation(result, signal_ok)
    elif request_category == "Emergency_Response_Request":
        _approve_emergency_response(result, vehicle_type, kb_priority, corridor_ok)
    elif request_category == "Integrated_City_Service_Request":
        _approve_integrated_service(result, vehicle_type, kb_priority, corridor_ok, signal_ok)

    return result
