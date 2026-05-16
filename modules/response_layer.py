"""
=============================================================================
Module: response_layer.py
Description:
    Final Response Layer for Smart City Traffic & Emergency Response
    AI System.
    Aggregates outputs from all active modules (ANN, Knowledge Base,
    CSP, Search) into a single structured, readable final response.
=============================================================================
"""

REPORT_SEPARATOR = "=" * 65
THIN_SEPARATOR = "-" * 65


def _make_base_response(request: dict, pipeline_outputs: dict) -> dict:
    """Create the standard final response structure."""
    return {
        "request_id": request.get("request_id", "N/A"),
        "vehicle_type": request.get("vehicle_type", "N/A").title(),
        "request_category": request.get("request_category", "N/A"),
        "source": request.get("current_location", "N/A"),
        "destination": request.get("destination", "N/A"),
        "modules_used": list(pipeline_outputs.keys()),
        "decision": "PENDING",
        "summary": [],
        "details": {},
    }


def _add_ann_details(response: dict, ann: dict):
    """Attach ANN prediction details to the final response."""
    response["details"]["priority_prediction"] = {
        "predicted_priority": ann.get("priority_label", "N/A"),
        "confidence": f"{ann.get('confidence', 0)}%",
        "probabilities": ann.get("probabilities", {}),
    }
    response["summary"].append(
        f"ANN Priority Prediction: {ann.get('priority_label', 'N/A')} "
        f"(Confidence: {ann.get('confidence', 0)}%)"
    )


def _add_policy_details(response: dict, policy: dict) -> bool:
    """Attach policy details and return True if request remains approved."""
    approved = policy.get("approved", False)
    response["details"]["policy_validation"] = {
        "status": "APPROVED" if approved else "REJECTED",
        "kb_priority": policy.get("policy_priority", "N/A"),
        "signal_override": policy.get("signal_override_allowed", False),
        "emergency_corridor": policy.get("emergency_corridor", False),
        "rules_fired": policy.get("rules_fired", []),
        "explanation": " | ".join(policy.get("explanation", [])),
    }
    response["summary"].append(
        f"Policy Validation: {'APPROVED' if approved else 'REJECTED'} "
        f"(KB Priority: {policy.get('policy_priority', 'N/A')})"
    )
    if approved:
        return True

    response["decision"] = "REJECTED"
    response["summary"].append("REQUEST REJECTED by Knowledge Base policy.")
    response["decision_message"] = (
        "This request has been REJECTED. The vehicle is not authorized "
        "to perform the requested action under current traffic policy."
    )
    return False


def _add_csp_details(response: dict, csp: dict):
    """Attach CSP details to the final response."""
    if csp.get("success"):
        response["details"]["signal_control"] = {
            "control_zone": csp.get("control_zone", "N/A"),
            "assignment": csp.get("assignment", {}),
            "green_corridor": csp.get("green_corridor", []),
            "explanation": csp.get("explanation", ""),
        }
        response["summary"].append(
            f"Signal Control: {len(csp.get('green_corridor', []))} GREEN signals "
            f"assigned in {csp.get('control_zone', 'N/A')}."
        )
        return

    response["details"]["signal_control"] = {
        "status": "FAILED",
        "explanation": csp.get("explanation", "CSP failed."),
    }
    response["summary"].append("Signal Control: CSP allocation FAILED.")


def _add_search_details(response: dict, search: dict):
    """Attach search results to the final response."""
    if search.get("success"):
        path = search.get("path", [])
        response["details"]["route"] = {
            "algorithm": search.get("algorithm", "N/A"),
            "algorithm_reason": search.get("algorithm_reason", ""),
            "path": path,
            "route": " -> ".join(path),
            "cost": search.get("cost", 0),
            "hops": len(path) - 1,
            "explanation": search.get("explanation", ""),
        }
        response["summary"].append(
            f"Route ({search.get('algorithm', 'N/A')}): "
            f"{' -> '.join(path)} | Cost: {search.get('cost', 0)}"
        )
        return

    response["details"]["route"] = {
        "status": "NOT FOUND",
        "explanation": search.get("explanation", "No route found."),
    }
    response["summary"].append("Route: No valid path found.")


def _finalize_decision(response: dict, request: dict, pipeline_outputs: dict):
    """Set the final decision and message when request was not rejected."""
    if response["decision"] == "REJECTED":
        return
    if "search" in pipeline_outputs and not pipeline_outputs["search"].get("success"):
        response["decision"] = "PARTIAL"
        response["decision_message"] = "Request partially processed. Route could not be computed."
        return

    response["decision"] = "APPROVED"
    response["decision_message"] = _build_decision_message(request, pipeline_outputs)


def build_final_response(request: dict, pipeline_outputs: dict) -> dict:
    """
    -------------------------------------------------------------------------
    Function: build_final_response
    Description:
        Main entry point for the Response Layer.
    -------------------------------------------------------------------------
    """
    response = _make_base_response(request, pipeline_outputs)

    if "ann" in pipeline_outputs:
        _add_ann_details(response, pipeline_outputs["ann"])

    if "policy" in pipeline_outputs:
        if not _add_policy_details(response, pipeline_outputs["policy"]):
            return response

    if "csp" in pipeline_outputs:
        _add_csp_details(response, pipeline_outputs["csp"])

    if "search" in pipeline_outputs:
        _add_search_details(response, pipeline_outputs["search"])

    _finalize_decision(response, request, pipeline_outputs)
    return response


def _build_decision_message(request: dict, outputs: dict) -> str:
    """
    -------------------------------------------------------------------------
    Function: _build_decision_message
    Description:
        Builds a human-readable final decision message summarizing the
        outcome of all modules used for this specific request.
    -------------------------------------------------------------------------
    """
    vehicle = request.get("vehicle_type", "vehicle").title()
    source = request.get("current_location", "origin")
    destination = request.get("destination", "destination")
    request_category = request.get("request_category", "")
    parts = []

    if request_category == "Route_Request":
        if "search" in outputs and outputs["search"].get("success"):
            path = outputs["search"].get("path", [])
            algorithm = outputs["search"].get("algorithm", "Search")
            parts.append(
                f"Standard route computed for {vehicle} from {source} to {destination} "
                f"using {algorithm}. Path: {' -> '.join(path)}."
            )

    elif request_category == "Policy_Check":
        parts.append(
            f"Policy check for {vehicle} completed. "
            f"Authorization status confirmed by Knowledge Base."
        )

    elif request_category == "Control_Allocation_Request":
        if "csp" in outputs and outputs["csp"].get("success"):
            zone = outputs["csp"].get("control_zone", "")
            greens = outputs["csp"].get("green_corridor", [])
            parts.append(
                f"Signal control plan assigned for {zone}. "
                f"GREEN intersections: {', '.join(greens) if greens else 'None'}."
            )

    elif request_category == "Emergency_Response_Request":
        if "ann" in outputs:
            parts.append(
                f"Emergency priority [{outputs['ann'].get('priority_label', 'N/A')}] "
                f"predicted and validated."
            )
        if "search" in outputs and outputs["search"].get("success"):
            parts.append(f"Emergency route to {destination}: {' -> '.join(outputs['search'].get('path', []))}.")

    elif request_category == "Integrated_City_Service_Request":
        if "ann" in outputs:
            parts.append(f"Priority [{outputs['ann'].get('priority_label', 'N/A')}] confirmed.")
        if "csp" in outputs and outputs["csp"].get("success"):
            greens = outputs["csp"].get("green_corridor", [])
            parts.append(f"Corridor cleared: {', '.join(greens) if greens else 'N/A'}.")
        if "search" in outputs and outputs["search"].get("success"):
            parts.append(f"Optimal route: {' -> '.join(outputs['search'].get('path', []))}.")

    if not parts:
        parts.append(f"Request for {vehicle} from {source} to {destination} processed successfully.")

    return " ".join(parts)


def _format_probability_lines(lines: list, probabilities: dict):
    """Append probability details to the report output."""
    if probabilities:
        lines.append(
            f"    Probabilities      : Normal={probabilities.get('Normal', 0)}% | "
            f"High={probabilities.get('High', 0)}% | Critical={probabilities.get('Critical', 0)}%"
        )


def _format_rules(lines: list, rules: list):
    """Append the fired knowledge-base rules."""
    if not rules:
        return
    lines.append("    Rules Fired:")
    for rule in rules:
        lines.append(f"      - {rule}")


def _format_priority_section(lines: list, details: dict):
    if "priority_prediction" not in details:
        return
    priority = details["priority_prediction"]
    lines.append("  [ANN PRIORITY PREDICTION]")
    lines.append(f"    Predicted Priority : {priority.get('predicted_priority', 'N/A')}")
    lines.append(f"    Confidence         : {priority.get('confidence', 'N/A')}")
    _format_probability_lines(lines, priority.get("probabilities", {}))
    lines.append(THIN_SEPARATOR)


def _format_policy_section(lines: list, details: dict):
    if "policy_validation" not in details:
        return
    policy = details["policy_validation"]
    lines.append("  [POLICY VALIDATION]")
    lines.append(f"    Status          : {policy.get('status', 'N/A')}")
    lines.append(f"    KB Priority     : {policy.get('kb_priority', 'N/A')}")
    lines.append(f"    Signal Override : {'Yes' if policy.get('signal_override') else 'No'}")
    lines.append(f"    Emergency Corr. : {'Yes' if policy.get('emergency_corridor') else 'No'}")
    lines.append(f"    Explanation     : {policy.get('explanation', '')}")
    _format_rules(lines, policy.get("rules_fired", []))
    lines.append(THIN_SEPARATOR)


def _format_signal_section(lines: list, details: dict):
    if "signal_control" not in details:
        return
    signal = details["signal_control"]
    lines.append("  [SIGNAL CONTROL PLAN - CSP]")
    if "assignment" in signal:
        lines.append(f"    Control Zone    : {signal.get('control_zone', 'N/A')}")
        lines.append(f"    Green Corridor  : {', '.join(signal.get('green_corridor', []))}")
        lines.append("    Signal Assignment:")
        for intersection, state in signal.get("assignment", {}).items():
            lines.append(f"      {intersection}: {state}")
        lines.append(f"    Explanation     : {signal.get('explanation', '')}")
    else:
        lines.append(f"    Status: {signal.get('status', 'FAILED')}")
        lines.append(f"    {signal.get('explanation', '')}")
    lines.append(THIN_SEPARATOR)


def _format_route_section(lines: list, details: dict):
    if "route" not in details:
        return
    route = details["route"]
    lines.append("  [ROUTE / NAVIGATION]")
    if "path" in route:
        lines.append(f"    Algorithm       : {route.get('algorithm', 'N/A')}")
        lines.append(f"    Reason          : {route.get('algorithm_reason', '')}")
        lines.append(f"    Route           : {route.get('route', 'N/A')}")
        lines.append(f"    Total Cost      : {route.get('cost', 0)} units")
        lines.append(f"    Total Hops      : {route.get('hops', 0)}")
        lines.append(f"    Explanation     : {route.get('explanation', '')}")
    else:
        lines.append(f"    Status: {route.get('status', 'NOT FOUND')}")
        lines.append(f"    {route.get('explanation', '')}")
    lines.append(THIN_SEPARATOR)


def format_response_for_display(response: dict) -> str:
    """
    -------------------------------------------------------------------------
    Function: format_response_for_display
    Description:
        Converts the final response dict into a formatted, human-readable
        string for terminal display.
    -------------------------------------------------------------------------
    """
    lines = [
        REPORT_SEPARATOR,
        "       SMART CITY TRAFFIC & EMERGENCY RESPONSE AI SYSTEM",
        "                      FINAL RESPONSE REPORT",
        REPORT_SEPARATOR,
        f"  Request ID       : {response.get('request_id', 'N/A')}",
        f"  Vehicle Type     : {response.get('vehicle_type', 'N/A')}",
        f"  Request Category : {response.get('request_category', 'N/A')}",
        f"  Source           : {response.get('source', 'N/A')}",
        f"  Destination      : {response.get('destination', 'N/A')}",
        f"  Modules Used     : {', '.join(response.get('modules_used', []))}",
        THIN_SEPARATOR,
        f"  DECISION: {response.get('decision', 'PENDING')}",
        THIN_SEPARATOR,
    ]

    details = response.get("details", {})
    _format_priority_section(lines, details)
    _format_policy_section(lines, details)
    _format_signal_section(lines, details)
    _format_route_section(lines, details)

    lines.append("  [DECISION MESSAGE]")
    lines.append(f"    {response.get('decision_message', 'Request processed.')}")
    lines.append(REPORT_SEPARATOR)
    return "\n".join(lines)
