"""
=============================================================================
Module: router.py
Description:
    Request Router for Smart City Traffic & Emergency Response AI System.
    Acts as the control-flow manager. Determines which modules to invoke
    based on request_category and sequences them in the correct order.
=============================================================================
"""

from modules.ann_module import predict_priority
from modules.csp_module import solve_csp
from modules.knowledge_base import validate_request_policy
from modules.response_layer import build_final_response
from modules.search_module import find_route


def _is_emergency_vehicle(request: dict) -> bool:
    """Return True when the request vehicle is emergency-related."""
    return request["vehicle_type"] in ["ambulance", "fire_truck", "police"]


def _run_search(request: dict) -> dict:
    """Run the search module for the request route."""
    return find_route(request["current_location"], request["destination"], request)


def _run_policy(request: dict) -> dict:
    """Run the knowledge-base policy validation."""
    return validate_request_policy(request)


def _run_csp(request: dict, emergency_mode: bool, approved_action: str) -> dict:
    """Run the CSP module with the given action settings."""
    return solve_csp(
        request["control_zone"],
        emergency_mode=emergency_mode,
        approved_action=approved_action,
    )


def _route_standard_request(request: dict, outputs: dict):
    outputs["search"] = _run_search(request)


def _route_policy_check(request: dict, outputs: dict):
    outputs["policy"] = _run_policy(request)


def _route_control_allocation(request: dict, outputs: dict):
    outputs["policy"] = _run_policy(request)
    if outputs["policy"].get("approved"):
        is_emergency = _is_emergency_vehicle(request)
        outputs["csp"] = _run_csp(
            request,
            emergency_mode=is_emergency,
            approved_action="SignalOverride" if is_emergency else "LaneControl",
        )


def _route_emergency_request(request: dict, outputs: dict):
    outputs["ann"] = predict_priority(request["ann_features"])
    outputs["policy"] = _run_policy(request)
    if outputs["policy"].get("approved"):
        outputs["csp"] = _run_csp(
            request,
            emergency_mode=True,
            approved_action="EmergencyRoute",
        )
        outputs["search"] = _run_search(request)


def _route_integrated_service(request: dict, outputs: dict):
    outputs["ann"] = predict_priority(request["ann_features"])
    outputs["policy"] = _run_policy(request)
    if outputs["policy"].get("approved"):
        outputs["csp"] = _run_csp(
            request,
            emergency_mode=True,
            approved_action="CorridorAccess",
        )
        outputs["search"] = _run_search(request)


PIPELINE_HANDLERS = {
    "Route_Request": _route_standard_request,
    "Policy_Check": _route_policy_check,
    "Control_Allocation_Request": _route_control_allocation,
    "Emergency_Response_Request": _route_emergency_request,
    "Integrated_City_Service_Request": _route_integrated_service,
}


def route_request(request: dict) -> dict:
    """
    -------------------------------------------------------------------------
    Function: route_request
    Description:
        Selects and executes the correct processing pipeline based on the
        request_category field, then builds the final response.
    -------------------------------------------------------------------------
    """
    request_category = request.get("request_category", "")
    handler = PIPELINE_HANDLERS.get(request_category)

    if handler is None:
        return {
            "decision": "ERROR",
            "decision_message": f"Unknown request category: '{request_category}'",
            "modules_used": [],
        }

    outputs = {}
    handler(request, outputs)
    return build_final_response(request, outputs)
