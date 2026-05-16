"""
=============================================================================
Program : main.py
Project : Smart City Traffic & Emergency Response AI System
Course  : AL-2002 Artificial Intelligence Lab - Final Project
=============================================================================
Description:
    Main entry point and terminal-based interface for the Smart City Traffic
    & Emergency Response AI System. Provides an interactive menu allowing
    users to submit traffic requests, test the ANN priority predictor,
    view the city road network, and compare search algorithms.
=============================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ann_module import PRIORITY_LABELS, TRAINING_DATA, get_trained_model, predict_priority
from modules.csp_module import INTERSECTION_GRAPH, ZONE_INTERSECTIONS
from modules.preprocessing import (
    VALID_CONTROL_ZONES,
    VALID_LOCATIONS,
    VALID_REQUEST_CATEGORIES,
    VALID_SEVERITY_LEVELS,
    VALID_VEHICLE_TYPES,
    preprocess,
)
from modules.response_layer import format_response_for_display
from modules.router import route_request
from modules.search_module import CITY_GRAPH_WEIGHTED, astar, bfs, ucs

try:
    from modules.visualization import (
        show_algorithm_comparison,
        show_ann_accuracy,
        show_city_network_graph,
        show_csp_graph as show_csp_graph_visual,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


MENU_OPTIONS = [
    ("1", "Submit a New Traffic Request"),
    ("2", "Test ANN Priority Predictor"),
    ("3", "View City Road Network"),
    ("4", "View CSP Intersection Graph"),
    ("5", "Run All Algorithms Comparison (BFS vs UCS vs A*)"),
    ("0", "Exit"),
]

ANN_LABELS = {
    "Vehicle": "ambulance=0, fire_truck=1, police=2, civilian=3",
    "Severity": "low=0, medium=1, high=2, critical=3",
    "Time Sensitivity": "low=0, medium=1, high=2",
    "Traffic Density": "low=0, medium=1, high=2",
    "Estimated Distance": "1-10",
}


def clear():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    """Print the main system header banner."""
    print("=" * 65)
    print("    SMART CITY TRAFFIC & EMERGENCY RESPONSE AI SYSTEM")
    print("         AL-2002 Artificial Intelligence Lab")
    print("=" * 65)


def print_separator(width: int = 60):
    """Print a standard section separator."""
    print("  " + "-" * width)


def pause():
    """Wait for the user to press Enter."""
    input("\n  Press ENTER to continue...")


def print_menu():
    """Print the main menu."""
    print("\n  MAIN MENU")
    print("  -" * 30)
    for key, label in MENU_OPTIONS:
        print(f"  [{key}] {label}")
    print_separator()


def choose_from_list(prompt: str, options: list) -> str:
    """Display options and return the selected value."""
    print(f"\n  {prompt}")
    for index, option in enumerate(options, 1):
        print(f"    [{index}] {option}")

    while True:
        try:
            choice = int(input("  Enter number: ").strip()) - 1
            if 0 <= choice < len(options):
                return options[choice]
            print(f"  Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("  Invalid input. Please enter a number.")


def get_text_input(prompt: str, default: str = "") -> str:
    """Read text input and apply the default when allowed."""
    if default:
        value = input(f"  {prompt} [{default}]: ").strip()
        return value if value else default

    while True:
        value = input(f"  {prompt}: ").strip()
        if value:
            return value
        print("  This field cannot be empty.")


def _print_screen_title(title: str, subtitle: str = ""):
    """Clear the screen and print the shared page header."""
    clear()
    print_header()
    print(f"\n  {title}")
    print_separator()
    if subtitle:
        print(f"  {subtitle}")


def _safe_visualization(message: str, callback, *args, **kwargs):
    """Run a visualization callback only when plotting is available."""
    if not VISUALIZATION_AVAILABLE:
        print("\n  [!] Visualization not available (matplotlib/networkx missing).")
        return
    print(f"\n  [Graph] {message}")
    callback(*args, **kwargs)


def _build_request_input() -> dict:
    """Collect all request fields from the user."""
    request_id = get_text_input("Request ID (e.g. REQ-001)", "REQ-001")
    vehicle_type = choose_from_list("Select Vehicle Type:", VALID_VEHICLE_TYPES)
    request_category = choose_from_list("Select Request Category:", VALID_REQUEST_CATEGORIES)
    current_location = choose_from_list("Select Current Location:", VALID_LOCATIONS)
    destination = choose_from_list(
        "Select Destination:",
        [location for location in VALID_LOCATIONS if location != current_location],
    )
    incident_severity = choose_from_list("Incident Severity:", VALID_SEVERITY_LEVELS)
    time_sensitivity = choose_from_list("Time Sensitivity:", ["low", "medium", "high"])
    traffic_density = choose_from_list("Traffic Density:", ["low", "medium", "high"])
    control_zone = choose_from_list("Control Zone:", VALID_CONTROL_ZONES)
    description_note = get_text_input("Description Note (optional)", "No additional notes")

    return {
        "request_id": request_id,
        "vehicle_type": vehicle_type,
        "request_category": request_category,
        "current_location": current_location,
        "destination": destination,
        "incident_severity": incident_severity,
        "time_sensitivity": time_sensitivity,
        "traffic_density": traffic_density,
        "priority_claim": "high" if vehicle_type != "civilian" else "normal",
        "control_zone": control_zone,
        "description_note": description_note,
    }


def submit_request():
    """Collect a traffic request, process it, and display the final result."""
    _print_screen_title(
        "SUBMIT NEW TRAFFIC REQUEST",
        "Fill in the request details below.",
    )
    print("  (Defaults shown in brackets - press Enter to use default)\n")

    try:
        request_input = _build_request_input()
        print("\n  Processing request...")
        print("  [Preprocessing] Validating and normalizing input...")
        processed_request = preprocess(request_input)
        print("  [Router] Selecting processing pipeline...")
        response = route_request(processed_request)
        print("  [Response Layer] Generating final response...\n")
        print(format_response_for_display(response))
    except ValueError as error:
        print(f"\n  INPUT ERROR: {error}")
    except Exception as error:
        print(f"\n  SYSTEM ERROR: {error}")

    pause()


def _read_ann_features() -> list:
    """Read manual ANN feature input from the user."""
    print()
    for label, value_map in ANN_LABELS.items():
        print(f"  {label}: {value_map}")
    print()
    return [
        int(input("  Vehicle type   (0-3): ").strip()),
        int(input("  Severity       (0-3): ").strip()),
        int(input("  Time sens.     (0-2): ").strip()),
        int(input("  Traffic density(0-2): ").strip()),
        int(input("  Distance       (1-10): ").strip()),
    ]


def _show_ann_prediction():
    """Display ANN prediction for manually entered features."""
    try:
        result = predict_priority(_read_ann_features())
        probabilities = result["probabilities"]
        print("\n  ANN PREDICTION RESULT")
        print("  " + "-" * 40)
        print(f"  Predicted Priority : {result['priority_label']}")
        print(f"  Confidence         : {result['confidence']}%")
        print(
            "  Probabilities      : "
            f"Normal={probabilities['Normal']}% | "
            f"High={probabilities['High']}% | "
            f"Critical={probabilities['Critical']}%"
        )
    except ValueError as error:
        print(f"\n  Invalid input: {error}")


def _show_ann_accuracy_report(model):
    """Display ANN accuracy on the training data."""
    print("\n  ACCURACY TEST ON TRAINING DATA")
    print("  " + "-" * 50)
    correct = 0
    total = len(TRAINING_DATA)

    for index, (features, true_label) in enumerate(TRAINING_DATA, 1):
        prediction = model.predict(features)
        status = "OK" if prediction == true_label else "WRONG"
        if prediction == true_label:
            correct += 1
        print(
            f"  [{index:2d}] Features={features} | "
            f"True={PRIORITY_LABELS[true_label]:8s} | "
            f"Pred={PRIORITY_LABELS[prediction]:8s} | {status}"
        )

    accuracy = (correct / total) * 100
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1f}%")
    _safe_visualization("Opening ANN accuracy chart...", show_ann_accuracy, TRAINING_DATA, model, PRIORITY_LABELS)


def test_ann():
    """Open the ANN test panel."""
    _print_screen_title("ANN PRIORITY PREDICTOR - TEST PANEL")
    print("  The ANN predicts priority: Normal / High / Critical")
    print("  Features: vehicle_type, severity, time_sensitivity, density, distance\n")
    print("  [1] Enter custom features manually")
    print("  [2] Run accuracy test on all training examples")
    print("  [0] Back")
    print_separator()

    choice = input("  Select: ").strip()
    if choice == "0":
        return

    model = get_trained_model()
    if choice == "1":
        _show_ann_prediction()
    elif choice == "2":
        _show_ann_accuracy_report(model)
    pause()


def show_city_network():
    """Print the city road network and optionally open the graph view."""
    _print_screen_title("CITY ROAD NETWORK (Weighted Graph)")
    print("  Format: Node --> Neighbor (cost)\n")
    for node, neighbors in CITY_GRAPH_WEIGHTED.items():
        neighbor_text = ", ".join(f"{neighbor} ({cost})" for neighbor, cost in neighbors)
        print(f"  {node:<20} --> {neighbor_text}")

    print("\n  Locations available:")
    for location in VALID_LOCATIONS:
        print(f"    - {location}")

    _safe_visualization("Opening visual road network window...", show_city_network_graph)
    pause()


def _print_zone_assignments(assignment: dict, zone: str):
    """Print the CSP assignment for a selected zone."""
    print(f"\n  Signal assignment for {zone}:")
    for intersection, state in assignment.items():
        print(f"    {intersection}: {state}")


def show_csp_graph():
    """Print CSP information and optionally open the graph view."""
    from modules.csp_module import solve_csp

    _print_screen_title("CSP INTERSECTION CONFLICT GRAPH")
    print("  Conflicting intersections (cannot both be GREEN):\n")
    for intersection, conflicts in INTERSECTION_GRAPH.items():
        print(f"  {intersection} conflicts with: {', '.join(conflicts)}")

    print("\n  Control Zone -> Intersections Mapping:\n")
    for zone, intersections in ZONE_INTERSECTIONS.items():
        print(f"  {zone}: {', '.join(intersections)}")

    if not VISUALIZATION_AVAILABLE:
        print("\n  [!] Visualization not available (matplotlib/networkx missing).")
        pause()
        return

    print("\n  [Graph] Select a zone to visualize its signal assignment:")
    zones = list(ZONE_INTERSECTIONS.keys())
    for index, zone in enumerate(zones, 1):
        print(f"    [{index}] {zone}")
    print("    [0] Show plain conflict graph (no assignment)")

    selection = input("  Select: ").strip()
    if selection == "0":
        show_csp_graph_visual()
        pause()
        return

    try:
        selected_index = int(selection) - 1
        if not 0 <= selected_index < len(zones):
            print("  Invalid choice.")
            pause()
            return
        zone = zones[selected_index]
        result = solve_csp(zone)
        assignment = result.get("assignment", {})
        _print_zone_assignments(assignment, zone)
        show_csp_graph_visual(assignment=assignment, control_zone=zone)
    except ValueError:
        print("  Invalid input.")

    pause()


def _search_all_algorithms(source: str, destination: str) -> tuple:
    """Run BFS, UCS, and A* for the same route."""
    bfs_result = bfs(source, destination)
    ucs_result = ucs(source, destination)
    astar_result = astar(source, destination)
    return bfs_result, ucs_result, astar_result


def _print_algorithm_summary(results: dict):
    """Print the compact comparison table."""
    print(f"  {'Algorithm':<10} {'Path':<45} {'Cost':<8} {'Hops'}")
    print("  " + "-" * 80)
    for algorithm, result in results.items():
        if result["success"]:
            path_text = " -> ".join(result["path"])
            if len(path_text) > 43:
                path_text = path_text[:40] + "..."
            print(
                f"  {algorithm:<10} {path_text:<45} "
                f"{result.get('cost', 0):<8} {len(result.get('path', [])) - 1}"
            )
        else:
            print(f"  {algorithm:<10} {'NO PATH FOUND':<45} {'N/A':<8} N/A")


def _print_algorithm_details(results: dict):
    """Print detailed results for each algorithm."""
    print("\n  Detailed Results:")
    print_separator()
    for algorithm, result in results.items():
        print(f"\n  [{algorithm.strip()}]")
        print(f"    Path        : {' -> '.join(result.get('path', ['N/A']))}")
        print(f"    Cost        : {result.get('cost', 'N/A')}")
        print(f"    Explanation : {result.get('explanation', '')}")


def algorithm_comparison():
    """Compare BFS, UCS, and A* for a selected route."""
    _print_screen_title("SEARCH ALGORITHM COMPARISON (BFS vs UCS vs A*)")
    source = choose_from_list("Select Source:", VALID_LOCATIONS)
    destination = choose_from_list(
        "Select Destination:",
        [location for location in VALID_LOCATIONS if location != source],
    )

    print(f"\n  Running BFS, UCS, A* from '{source}' to '{destination}'...")
    print_separator()

    bfs_result, ucs_result, astar_result = _search_all_algorithms(source, destination)
    results = {
        "BFS": bfs_result,
        "UCS": ucs_result,
        "A* ": astar_result,
    }

    _print_algorithm_summary(results)
    _print_algorithm_details(results)

    _safe_visualization(
        "Opening algorithm comparison chart...",
        show_algorithm_comparison,
        source,
        destination,
        bfs_result,
        ucs_result,
        astar_result,
    )
    if astar_result["success"]:
        _safe_visualization(
            "Opening A* highlighted road network...",
            show_city_network_graph,
            astar_result["path"],
            "A*",
        )

    pause()


def _exit_program():
    """Exit the application."""
    clear()
    print_header()
    print("\n  Thank you for using Smart City Traffic & Emergency Response AI System.")
    print("  Exiting...\n")
    sys.exit(0)


MENU_ACTIONS = {
    "1": submit_request,
    "2": test_ann,
    "3": show_city_network,
    "4": show_csp_graph,
    "5": algorithm_comparison,
    "0": _exit_program,
}


def main():
    """Run the terminal application loop."""
    print("\n  Initializing AI system...")
    print("  Training ANN model on sample data...")
    get_trained_model()
    print("  System ready.\n")

    while True:
        clear()
        print_header()
        print_menu()
        choice = input("  Enter your choice: ").strip()
        action = MENU_ACTIONS.get(choice)

        if action:
            action()
        else:
            print("  Invalid choice. Please enter 0-5.")
            pause()


if __name__ == "__main__":
    main()
