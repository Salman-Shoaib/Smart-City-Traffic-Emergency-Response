"""
=============================================================================
Module: visualization.py
Description:
    Visualization Module for Smart City Traffic & Emergency Response AI System.
    Uses matplotlib and networkx to render:
      1. City Road Network (weighted graph)
      2. CSP Intersection Conflict Graph
      3. Search Algorithm Comparison (bar chart)
      4. ANN Training Accuracy (bar chart)
=============================================================================
"""

import matplotlib

# Try best interactive backends in order
for _backend in ("TkAgg", "Qt5Agg", "WXAgg", ""):
    try:
        if _backend:
            matplotlib.use(_backend)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.close()
        break
    except Exception:
        continue
else:
    import matplotlib.pyplot as plt  # Last resort default

import matplotlib.patches as mpatches
import networkx as nx


def _style_axes(ax):
    """Apply the shared dark theme to a chart axis."""
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444477")
    ax.tick_params(colors="#aaaacc")


def _make_figure(title: str, figsize=(10, 7)):
    """Create a styled figure with dark background."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_title(title, color="#e0e0ff", fontsize=14, fontweight="bold", pad=15)
    _style_axes(ax)
    return fig, ax


def _label_bars(ax, bars, values, offset: float, suffix: str = ""):
    """Draw text labels above bars."""
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value}{suffix}",
            ha="center",
            color="white",
            fontsize=12,
            fontweight="bold"
        )


def _build_city_graph():
    """Create the city road graph with weighted edges."""
    from modules.search_module import CITY_GRAPH_WEIGHTED

    graph = nx.Graph()
    for node, neighbours in CITY_GRAPH_WEIGHTED.items():
        for neighbor, cost in neighbours:
            graph.add_edge(node, neighbor, weight=cost)
    return graph


def _build_intersection_graph():
    """Create the CSP conflict graph."""
    from modules.csp_module import INTERSECTION_GRAPH

    graph = nx.Graph()
    for node, neighbours in INTERSECTION_GRAPH.items():
        for neighbor in neighbours:
            graph.add_edge(node, neighbor)
    return graph


# ---------------------------------------------------------------------------
# 1. City Road Network Graph
# ---------------------------------------------------------------------------

CITY_NODE_POS = {
    "Central_Junction": (0, 0),
    "North_Gate": (0, 3),
    "South_Gate": (0, -3),
    "East_Avenue": (3, 0),
    "West_Boulevard": (-3, 0),
    "Market_Square": (2, -2),
    "City_Hospital": (-1, 5),
    "Fire_Station": (-3, 3),
    "Police_HQ": (1, -5),
    "Industrial_Zone": (4, -3),
}

CITY_LABELS = {
    "Central_Junction": "Central\nJunction",
    "North_Gate": "North\nGate",
    "South_Gate": "South\nGate",
    "East_Avenue": "East\nAvenue",
    "West_Boulevard": "West\nBlvd",
    "Market_Square": "Market\nSquare",
    "City_Hospital": "City\nHospital",
    "Fire_Station": "Fire\nStation",
    "Police_HQ": "Police\nHQ",
    "Industrial_Zone": "Industrial\nZone",
}

CITY_NODE_COLORS = {
    "Central_Junction": "#f39c12",
    "North_Gate": "#3498db",
    "South_Gate": "#3498db",
    "East_Avenue": "#3498db",
    "West_Boulevard": "#3498db",
    "Market_Square": "#9b59b6",
    "City_Hospital": "#e74c3c",
    "Fire_Station": "#e67e22",
    "Police_HQ": "#1abc9c",
    "Industrial_Zone": "#95a5a6",
}


def show_city_network_graph(highlight_path: list = None, algorithm_name: str = ""):
    """
    -------------------------------------------------------------------------
    Function: show_city_network_graph
    Description:
        Renders the weighted city road network using NetworkX and matplotlib.
        Optionally highlights a found path in bright green.
        highlight_path: list of node names forming the route (or None).
        algorithm_name: label shown in title when a path is highlighted.
    -------------------------------------------------------------------------
    """
    graph = _build_city_graph()
    title = "City Road Network - Weighted Graph"
    if highlight_path and algorithm_name:
        title = f"City Road Network - {algorithm_name} Route Highlighted"

    fig, ax = _make_figure(title, figsize=(13, 8))

    nx.draw_networkx_edges(
        graph,
        CITY_NODE_POS,
        ax=ax,
        edge_color="#334466",
        width=1.8,
        alpha=0.7
    )

    if highlight_path and len(highlight_path) > 1:
        path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
        nx.draw_networkx_edges(
            graph,
            CITY_NODE_POS,
            edgelist=path_edges,
            ax=ax,
            edge_color="#00ff88",
            width=4.0,
            alpha=1.0,
            style="solid"
        )

    base_colors = [CITY_NODE_COLORS.get(node, "#aaaaaa") for node in graph.nodes()]
    nx.draw_networkx_nodes(
        graph,
        CITY_NODE_POS,
        ax=ax,
        node_color=base_colors,
        node_size=900,
        alpha=0.95
    )

    if highlight_path:
        path_colors = []
        for node in graph.nodes():
            if node == highlight_path[0]:
                path_colors.append("#00ff88")
            elif node == highlight_path[-1]:
                path_colors.append("#ff4444")
            elif node in highlight_path:
                path_colors.append("#88ffaa")
            else:
                path_colors.append(CITY_NODE_COLORS.get(node, "#aaaaaa"))

        nx.draw_networkx_nodes(
            graph,
            CITY_NODE_POS,
            ax=ax,
            node_color=path_colors,
            node_size=900,
            alpha=0.95
        )

    nx.draw_networkx_labels(
        graph,
        CITY_NODE_POS,
        labels=CITY_LABELS,
        ax=ax,
        font_color="white",
        font_size=6.5,
        font_weight="bold"
    )

    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(
        graph,
        CITY_NODE_POS,
        edge_labels=edge_labels,
        ax=ax,
        font_color="#ffcc44",
        font_size=7,
        bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec="none", alpha=0.7)
    )

    legend_items = [
        mpatches.Patch(color="#f39c12", label="Central Junction"),
        mpatches.Patch(color="#3498db", label="Gate / Avenue"),
        mpatches.Patch(color="#e74c3c", label="Hospital"),
        mpatches.Patch(color="#e67e22", label="Fire Station"),
        mpatches.Patch(color="#1abc9c", label="Police HQ"),
        mpatches.Patch(color="#9b59b6", label="Market Square"),
        mpatches.Patch(color="#95a5a6", label="Industrial Zone"),
    ]
    if highlight_path:
        legend_items.extend([
            mpatches.Patch(color="#00ff88", label="Start"),
            mpatches.Patch(color="#ff4444", label="End"),
            mpatches.Patch(color="#88ffaa", label="Path Node"),
        ])

    ax.legend(
        handles=legend_items,
        loc="upper right",
        facecolor="#1a1a2e",
        edgecolor="#444477",
        labelcolor="white",
        fontsize=8
    )

    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. CSP Intersection Conflict Graph
# ---------------------------------------------------------------------------

CSP_COLORS = {
    "GREEN": "#2ecc71",
    "YELLOW": "#f1c40f",
    "RED": "#e74c3c",
    None: "#778899",
}


def show_csp_graph(assignment: dict = None, control_zone: str = ""):
    """
    -------------------------------------------------------------------------
    Function: show_csp_graph
    Description:
        Renders the CSP intersection conflict graph using NetworkX.
        If an assignment dict is provided (INT -> signal state) the nodes
        are coloured GREEN / YELLOW / RED accordingly.
        assignment: dict like {"INT_1": "GREEN", ...} or None for plain view.
        control_zone: label shown in title.
    -------------------------------------------------------------------------
    """
    graph = _build_intersection_graph()
    title = "CSP Intersection Conflict Graph"
    if control_zone:
        title += f" - {control_zone}"

    fig, ax = _make_figure(title, figsize=(9, 7))
    pos = nx.spring_layout(graph, seed=42, k=1.5)

    if assignment:
        node_colors = [CSP_COLORS.get(assignment.get(node), "#778899") for node in graph.nodes()]
    else:
        node_colors = ["#5588cc"] * len(graph.nodes())

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="#cc4444",
        width=2.0,
        style="dashed",
        alpha=0.8
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=1200,
        edgecolors="white",
        linewidths=1.5
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        font_color="white",
        font_size=10,
        font_weight="bold"
    )

    legend_items = [
        mpatches.Patch(color="#cc4444", label="Conflict Edge (no both GREEN)"),
        mpatches.Patch(color="#2ecc71", label="GREEN signal"),
        mpatches.Patch(color="#f1c40f", label="YELLOW signal"),
        mpatches.Patch(color="#e74c3c", label="RED signal"),
        mpatches.Patch(color="#778899", label="Unassigned"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper left",
        facecolor="#1a1a2e",
        edgecolor="#444477",
        labelcolor="white",
        fontsize=9
    )

    if assignment:
        info = "  ".join([f"{key}: {value}" for key, value in sorted(assignment.items())])
        fig.text(0.5, 0.02, info, ha="center", color="#aaaacc", fontsize=8)

    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Algorithm Comparison Bar Chart
# ---------------------------------------------------------------------------

def show_algorithm_comparison(
    source: str,
    destination: str,
    bfs_result: dict,
    ucs_result: dict,
    astar_result: dict
):
    """
    -------------------------------------------------------------------------
    Function: show_algorithm_comparison
    Description:
        Draws a side-by-side bar chart comparing BFS, UCS, and A* results:
        - Path cost (or hops for BFS)
        - Number of hops
        Also prints the paths below the chart.
    -------------------------------------------------------------------------
    """
    algorithms = ["BFS", "UCS", "A*"]
    results = [bfs_result, ucs_result, astar_result]
    colors = ["#3498db", "#e67e22", "#2ecc71"]

    costs = []
    hops = []
    for result in results:
        if result["success"]:
            costs.append(result.get("cost", 0))
            hops.append(len(result.get("path", [])) - 1)
        else:
            costs.append(0)
            hops.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"Search Algorithm Comparison\n{source} -> {destination}",
        color="#e0e0ff",
        fontsize=13,
        fontweight="bold"
    )

    bar_kw = dict(width=0.45, edgecolor="white", linewidth=0.8)

    ax1 = axes[0]
    _style_axes(ax1)
    cost_bars = ax1.bar(algorithms, costs, color=colors, **bar_kw)
    ax1.set_title("Total Path Cost (units)", color="#aaaacc", fontsize=11)
    ax1.set_ylabel("Cost", color="#aaaacc")
    _label_bars(ax1, cost_bars, costs, offset=0.3)

    ax2 = axes[1]
    _style_axes(ax2)
    hop_bars = ax2.bar(algorithms, hops, color=colors, **bar_kw)
    ax2.set_title("Number of Hops (edges)", color="#aaaacc", fontsize=11)
    ax2.set_ylabel("Hops", color="#aaaacc")
    _label_bars(ax2, hop_bars, hops, offset=0.05)

    path_texts = []
    for name, result in zip(algorithms, results):
        if result["success"]:
            path_texts.append(f"{name}: {' -> '.join(result['path'])}")
        else:
            path_texts.append(f"{name}: No path found")

    fig.text(
        0.5,
        0.01,
        "\n".join(path_texts),
        ha="center",
        color="#aaaacc",
        fontsize=7.5,
        verticalalignment="bottom"
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.show()


# ---------------------------------------------------------------------------
# 4. ANN Training Accuracy Bar Chart
# ---------------------------------------------------------------------------

def show_ann_accuracy(training_data: list, model, priority_labels: dict):
    """
    -------------------------------------------------------------------------
    Function: show_ann_accuracy
    Description:
        Runs the trained ANN model over all training examples and plots:
        - Per-class accuracy (Normal / High / Critical)
        - Overall accuracy
        - Individual prediction correctness as a scatter strip chart.
    -------------------------------------------------------------------------
    """
    results_by_class = {
        0: {"correct": 0, "total": 0},
        1: {"correct": 0, "total": 0},
        2: {"correct": 0, "total": 0},
    }
    all_correct = []

    for features, true_label in training_data:
        prediction = model.predict(features)
        is_correct = prediction == true_label
        results_by_class[true_label]["total"] += 1
        results_by_class[true_label]["correct"] += int(is_correct)
        all_correct.append(is_correct)

    class_names = [priority_labels[key] for key in sorted(priority_labels)]
    class_accs = [
        (results_by_class[key]["correct"] / results_by_class[key]["total"]) * 100
        if results_by_class[key]["total"] > 0 else 0
        for key in sorted(results_by_class)
    ]
    overall = sum(all_correct) / len(all_correct) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"ANN Priority Predictor - Training Accuracy  (Overall: {overall:.1f}%)",
        color="#e0e0ff",
        fontsize=13,
        fontweight="bold"
    )

    bar_colors = ["#3498db", "#e67e22", "#e74c3c"]

    ax1 = axes[0]
    _style_axes(ax1)
    bars = ax1.bar(
        class_names,
        class_accs,
        color=bar_colors,
        width=0.45,
        edgecolor="white",
        linewidth=0.8
    )
    ax1.set_ylim(0, 110)
    ax1.axhline(100, color="#aaaacc", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_title("Per-Class Accuracy (%)", color="#aaaacc", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", color="#aaaacc")
    _label_bars(
        ax1,
        bars,
        [f"{value:.1f}" for value in class_accs],
        offset=1,
        suffix="%"
    )

    ax2 = axes[1]
    _style_axes(ax2)
    for index, (features, true_label) in enumerate(training_data):
        prediction = model.predict(features)
        is_correct = prediction == true_label
        color = "#2ecc71" if is_correct else "#e74c3c"
        marker = "o" if is_correct else "x"
        ax2.scatter(index + 1, true_label, c=color, marker=marker, s=120, zorder=3, linewidths=2)

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(class_names, color="#aaaacc")
    ax2.set_xlabel("Training Sample #", color="#aaaacc")
    ax2.set_title("Prediction per Sample  (O Correct / X Wrong)", color="#aaaacc", fontsize=11)
    ax2.grid(axis="y", color="#334466", linestyle="--", alpha=0.5)

    correct_patch = mpatches.Patch(color="#2ecc71", label="Correct")
    wrong_patch = mpatches.Patch(color="#e74c3c", label="Wrong")
    ax2.legend(
        handles=[correct_patch, wrong_patch],
        facecolor="#1a1a2e",
        edgecolor="#444477",
        labelcolor="white",
        fontsize=9
    )

    plt.tight_layout()
    plt.show()
