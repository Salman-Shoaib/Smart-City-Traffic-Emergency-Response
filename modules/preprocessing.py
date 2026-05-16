"""
=============================================================================
Module: preprocessing.py
Description:
    Input & Preprocessing Module for Smart City Traffic & Emergency Response
    AI System. This module validates incoming traffic requests, normalizes
    field values, maps categorical data, and builds the internal request
    object. It also prepares the ANN feature vector for priority prediction.
=============================================================================
"""

VALID_VEHICLE_TYPES = ["ambulance", "fire_truck", "police", "civilian"]
VALID_REQUEST_CATEGORIES = [
    "Route_Request",
    "Policy_Check",
    "Control_Allocation_Request",
    "Emergency_Response_Request",
    "Integrated_City_Service_Request",
]
VALID_SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
VALID_LOCATIONS = [
    "Central_Junction",
    "North_Gate",
    "South_Gate",
    "East_Avenue",
    "West_Boulevard",
    "City_Hospital",
    "Fire_Station",
    "Police_HQ",
    "Market_Square",
    "Industrial_Zone",
]
VALID_CONTROL_ZONES = ["Zone_A", "Zone_B", "Zone_C", "Zone_D", "Zone_E"]
VALID_PRIORITY_LEVELS = ["normal", "high"]
VALID_THREE_LEVELS = ["low", "medium", "high"]

VEHICLE_MAP = {"ambulance": 0, "fire_truck": 1, "police": 2, "civilian": 3}
SEVERITY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SENSITIVITY_MAP = {"low": 0, "medium": 1, "high": 2}
DENSITY_MAP = {"low": 0, "medium": 1, "high": 2}

DISTANCE_MAP = {
    ("Central_Junction", "City_Hospital"): 5,
    ("Central_Junction", "North_Gate"): 3,
    ("Central_Junction", "South_Gate"): 3,
    ("Central_Junction", "East_Avenue"): 4,
    ("Central_Junction", "West_Boulevard"): 4,
    ("Central_Junction", "Market_Square"): 2,
    ("North_Gate", "City_Hospital"): 6,
    ("North_Gate", "Fire_Station"): 4,
    ("South_Gate", "Police_HQ"): 3,
    ("East_Avenue", "Industrial_Zone"): 5,
    ("West_Boulevard", "Market_Square"): 3,
}


def _require_field(raw_input: dict, field_name: str):
    """Return a required field value or raise a clear error."""
    value = raw_input.get(field_name, "")
    if str(value).strip() == "":
        raise ValueError(f"Missing required field: '{field_name}'")
    return value


def _normalize_text(value, lower: bool = False) -> str:
    """Convert a value to stripped text and optionally lowercase it."""
    text = str(value).strip()
    return text.lower() if lower else text


def _validate_choice(field_name: str, value: str, valid_options: list):
    """Raise ValueError if value is not in valid_options."""
    if value not in valid_options:
        raise ValueError(
            f"Invalid {field_name} '{value}'.\n"
            f"Valid options: {valid_options}"
        )


def _get_distance(source: str, destination: str) -> int:
    """Return a simple estimated distance used by the ANN."""
    return DISTANCE_MAP.get((source, destination), DISTANCE_MAP.get((destination, source), 4))


def _build_ann_features(cleaned: dict, estimated_distance: int) -> list:
    """Build the ANN feature list in the exact order expected by the model."""
    return [
        VEHICLE_MAP.get(cleaned["vehicle_type"], 3),
        SEVERITY_MAP.get(cleaned["incident_severity"], 0),
        SENSITIVITY_MAP.get(cleaned["time_sensitivity"], 0),
        DENSITY_MAP.get(cleaned["traffic_density"], 0),
        estimated_distance,
    ]


def validate_request(raw_input: dict) -> dict:
    """
    -------------------------------------------------------------------------
    Function: validate_request
    Description:
        Validates all fields in the raw traffic request dictionary.
        Raises ValueError with a clear message if any field is missing
        or contains an invalid value. Returns a cleaned copy of the input.
    -------------------------------------------------------------------------
    """
    cleaned = {
        "request_id": _normalize_text(_require_field(raw_input, "request_id")),
        "vehicle_type": _normalize_text(_require_field(raw_input, "vehicle_type"), lower=True),
        "request_category": _normalize_text(_require_field(raw_input, "request_category")),
        "current_location": _normalize_text(_require_field(raw_input, "current_location")),
        "destination": _normalize_text(_require_field(raw_input, "destination")),
        "incident_severity": _normalize_text(raw_input.get("incident_severity", "low"), lower=True),
        "time_sensitivity": _normalize_text(raw_input.get("time_sensitivity", "low"), lower=True),
        "traffic_density": _normalize_text(raw_input.get("traffic_density", "low"), lower=True),
        "priority_claim": _normalize_text(raw_input.get("priority_claim", "normal"), lower=True),
        "control_zone": _normalize_text(raw_input.get("control_zone", "Zone_A")),
        "description_note": _normalize_text(raw_input.get("description_note", "")),
    }

    _validate_choice("vehicle_type", cleaned["vehicle_type"], VALID_VEHICLE_TYPES)
    _validate_choice("request_category", cleaned["request_category"], VALID_REQUEST_CATEGORIES)
    _validate_choice("current_location", cleaned["current_location"], VALID_LOCATIONS)
    _validate_choice("destination", cleaned["destination"], VALID_LOCATIONS)
    _validate_choice("incident_severity", cleaned["incident_severity"], VALID_SEVERITY_LEVELS)
    _validate_choice("control_zone", cleaned["control_zone"], VALID_CONTROL_ZONES)

    if cleaned["time_sensitivity"] not in VALID_THREE_LEVELS:
        raise ValueError("time_sensitivity must be 'low', 'medium', or 'high'.")
    if cleaned["traffic_density"] not in VALID_THREE_LEVELS:
        raise ValueError("traffic_density must be 'low', 'medium', or 'high'.")
    if cleaned["priority_claim"] not in VALID_PRIORITY_LEVELS:
        cleaned["priority_claim"] = "normal"
    if cleaned["current_location"] == cleaned["destination"]:
        raise ValueError("current_location and destination cannot be the same.")

    return cleaned


def normalize_request(cleaned: dict) -> dict:
    """
    -------------------------------------------------------------------------
    Function: normalize_request
    Description:
        Normalizes and derives additional internal fields from the cleaned
        request. Maps categorical text values to numeric codes used by the
        ANN module, and adds derived fields such as is_emergency and
        estimated_distance.
    -------------------------------------------------------------------------
    """
    estimated_distance = _get_distance(
        cleaned["current_location"],
        cleaned["destination"],
    )

    normalized = dict(cleaned)
    normalized["is_emergency"] = cleaned["vehicle_type"] != "civilian"
    normalized["estimated_distance"] = estimated_distance
    normalized["ann_features"] = _build_ann_features(cleaned, estimated_distance)
    return normalized


def preprocess(raw_input: dict) -> dict:
    """
    -------------------------------------------------------------------------
    Function: preprocess
    Description:
        Main entry point for the preprocessing module.
        Calls validate_request then normalize_request and returns the
        final internal request object ready for routing.
    -------------------------------------------------------------------------
    """
    return normalize_request(validate_request(raw_input))
