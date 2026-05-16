# Smart City Traffic & Emergency Response AI System

An AI-powered smart city simulation system that combines multiple Artificial Intelligence paradigms into one integrated architecture for traffic management and emergency response handling.

This project demonstrates how different AI techniques can work together inside a real-world inspired intelligent workflow.

---

# Project Overview

The system simulates a smart city environment capable of:

* Managing intelligent traffic routing
* Handling emergency response requests
* Predicting request urgency and priority
* Validating traffic rules and permissions
* Controlling traffic signals safely
* Generating optimized routes
* Producing explainable AI-based responses

The architecture combines:

* Artificial Neural Networks (ANN)
* Rule-Based Expert Systems
* Predicate Logic
* Constraint Satisfaction Problems (CSP)
* Search Algorithms
* Graph Theory
* Heuristic Search
* Explainable AI

---

# Complete System Workflow

```text
User Request
     ↓
Input & Preprocessing
     ↓
Request Router
     ↓
AI Modules
(ANN / Knowledge Base / CSP / Search)
     ↓
Final Response Layer
```

---

# AI Concepts Used

| AI Concept                            | Module            | Purpose                     |
| ------------------------------------- | ----------------- | --------------------------- |
| Artificial Neural Network (ANN)       | ann_module.py     | Predict request urgency     |
| Multi-Layer Perceptron (MLP)          | ann_module.py     | Neural network architecture |
| Rule-Based Expert System              | knowledge_base.py | Policy validation           |
| Predicate Logic                       | knowledge_base.py | Logical reasoning           |
| Constraint Satisfaction Problem (CSP) | csp_module.py     | Traffic signal assignment   |
| BFS                                   | search_module.py  | Shortest-hop routing        |
| UCS                                   | search_module.py  | Lowest-cost routing         |
| A* Search                             | search_module.py  | Fast emergency routing      |
| Graph Theory                          | search_module.py  | Road network modeling       |
| Heuristic Search                      | search_module.py  | Intelligent path estimation |
| Backtracking                          | csp_module.py     | Constraint solving          |
| Explainable AI                        | response_layer.py | Human-readable outputs      |

---

# Project Structure

```text
Project/
│
├── main.py
│
├── modules/
│   ├── preprocessing.py
│   ├── router.py
│   ├── ann_module.py
│   ├── knowledge_base.py
│   ├── csp_module.py
│   ├── search_module.py
│   ├── response_layer.py
│   ├── visualization.py
│   └── __init__.py
```

---

# Module Explanation

## 1. preprocessing.py

### Purpose

Handles input validation, normalization, and feature extraction.

### Main Responsibilities

* Validate user inputs
* Normalize text and request fields
* Convert categorical values into numerical ANN features
* Generate ANN feature vectors

### Concepts Used

* Data Validation
* Feature Engineering
* Label Encoding
* Data Normalization
* Structured Input Processing

### Example Feature Vector

```python
[0, 2, 2, 2, 5]
```

Where:

* `0` → Ambulance
* `2` → High severity
* `2` → High sensitivity
* `2` → High traffic density
* `5` → Estimated distance

---

## 2. router.py

### Purpose

Acts as the control flow manager of the system.

### Responsibilities

* Select the correct AI pipeline
* Route requests dynamically
* Coordinate module execution

### Pipeline Examples

| Request Type               | Modules Used            |
| -------------------------- | ----------------------- |
| Route Request              | Search Module           |
| Policy Check               | Knowledge Base          |
| Emergency Response Request | ANN + KB + CSP + Search |

### Concepts Used

* Pipeline-Based Architecture
* Dynamic Dispatch
* Modular AI Workflow

---

## 3. ann_module.py

### Purpose

Predicts urgency and priority using a custom Artificial Neural Network.

### Neural Network Architecture

```text
Input Layer (5)
      ↓
Hidden Layer 1 (8)
      ↓
Hidden Layer 2 (6)
      ↓
Output Layer (3)
```

### Output Classes

* 0 → Normal
* 1 → High
* 2 → Critical

### Important Feature

The ANN is implemented completely from scratch without using:

* TensorFlow
* PyTorch
* scikit-learn

### Implemented Manually

* Forward Propagation
* Backpropagation
* Gradient Descent
* Activation Functions
* Weight Updates
* Softmax Output

### AI Concepts Used

* Artificial Neural Networks
* Multi-Layer Perceptron (MLP)
* Supervised Learning
* Gradient Descent
* Softmax Classification
* Feature Normalization

### A* Formula

The search system uses the following heuristic formula:

```math
f(n) = g(n) + h(n)
```

Where:

* `g(n)` = Actual path cost
* `h(n)` = Estimated remaining distance

---

## 4. knowledge_base.py

### Purpose

Implements rule-based reasoning and logical validation.

### Responsibilities

* Validate permissions
* Authorize emergency vehicles
* Apply traffic rules
* Prevent unsafe operations

### Concepts Used

* Predicate Logic
* Rule-Based Expert Systems
* Symbolic AI
* Inference Rules
* Safety Validation

### Example Rule

```text
IF vehicle is emergency
AND severity is high
THEN priority = Critical
```

### Important Role

Acts as the safety gatekeeper of the system.

Even if the ANN predicts high priority, all actions must still pass logical policy validation.

---

## 5. csp_module.py

### Purpose

Controls traffic signals using Constraint Satisfaction Problems.

### Responsibilities

* Assign signal states
* Prevent signal conflicts
* Create emergency green corridors

### Signal States

* GREEN
* YELLOW
* RED

### Core Constraint

Two connected intersections cannot both be GREEN simultaneously.

### Concepts Used

* Constraint Satisfaction Problem (CSP)
* Backtracking Search
* Constraint Checking
* Graph Coloring Style Logic
* State Assignment

### Emergency Mode

Emergency vehicles receive priority green corridors through intersections.

---

## 6. search_module.py

### Purpose

Handles intelligent route generation.

### Implemented Algorithms

| Algorithm | Purpose                |
| --------- | ---------------------- |
| BFS       | Minimum-hop routing    |
| UCS       | Lowest-cost routing    |
| A* Search | Fast emergency routing |

### Concepts Used

* Graph Search
* Priority Queues
* Heuristic Search
* Weighted Graphs
* Shortest Path Algorithms

### Intelligent Routing

| Situation         | Algorithm Selected |
| ----------------- | ------------------ |
| Civilian Request  | BFS                |
| Weighted Routing  | UCS                |
| Emergency Routing | A*                 |

---

## 7. response_layer.py

### Purpose

Combines outputs from all modules into a final explainable response.

### Responsibilities

* Aggregate AI results
* Generate structured reports
* Explain decisions clearly

### Concepts Used

* Explainable AI
* Decision Reporting
* Data Aggregation
* Context-Aware Reporting

### Example Output Information

* ANN priority prediction
* Confidence score
* Policy approvals
* Signal assignments
* Selected route
* Search algorithm used

---

## 8. visualization.py

### Purpose

Provides graph and chart visualizations.

### Libraries Used

* matplotlib
* networkx

### Visualizations

* City road network
* CSP conflict graph
* Search algorithm comparison
* ANN performance charts

### Concepts Used

* Data Visualization
* Network Graphs
* Comparative Analysis
* Graph Visualization

---

# End-to-End Emergency Request Flow

## Example Scenario

An ambulance requests routing from:

```text
Central_Junction → City_Hospital
```

Conditions:

* High severity
* High time sensitivity
* Heavy traffic

## Complete Flow

### Step 1 — Input Collection

User submits request.

### Step 2 — Preprocessing

System validates and standardizes data.

### Step 3 — Router

System selects:

```text
ANN → KB → CSP → Search
```

### Step 4 — ANN Prediction

ANN predicts:

```text
Critical Priority
```

### Step 5 — Knowledge Base

System validates:

* Signal override authorization
* Emergency corridor access
* Traffic rules

### Step 6 — CSP Allocation

Traffic signals are assigned safely.

### Step 7 — Search Module

A* finds the fastest emergency route.

### Step 8 — Final Response

System returns:

* Priority level
* Policy approvals
* Signal assignments
* Optimized route
* Explainable decisions

---

# Technologies Used

## Programming Language

* Python

## Python Libraries

* matplotlib
* networkx
* numpy

---

# Key Features

## Hybrid AI Architecture

Combines:

* Symbolic AI
* Machine Learning
* Search Algorithms
* Constraint Solving

inside one intelligent workflow.

## ANN Built From Scratch

The neural network implementation is completely manual for educational and academic purposes.

## Modular Design

Each module has a separate responsibility.

Advantages:

* Easier debugging
* Easier maintenance
* Better scalability
* Cleaner architecture
* Reusable components

## Explainable AI

The system explains:

* Why decisions were made
* Which modules were used
* Which rules were triggered
* Which algorithm generated the route

---

# Academic Topics Covered

This project demonstrates practical understanding of:

* Artificial Intelligence
* Machine Learning
* Neural Networks
* Expert Systems
* Predicate Logic
* Search Algorithms
* Constraint Satisfaction Problems
* Graph Theory
* Heuristic Search
* Explainable AI
* Software Architecture
* Data Validation
* Simulation Systems

---

# Example Menu

```text
1. Submit New Traffic Request
2. Test ANN Priority Predictor
3. View City Network
4. View CSP Graph
5. Compare Search Algorithms
0. Exit
```

---

# Example Emergency Request

```python
{
  "vehicle_type": "ambulance",
  "request_category": "Emergency_Response_Request",
  "current_location": "Central_Junction",
  "destination": "City_Hospital",
  "incident_severity": "high",
  "time_sensitivity": "high",
  "traffic_density": "high"
}
```

---

# Installation

## Clone Repository

```bash
git clone <repository-link>
cd <project-folder>
```

## Install Dependencies

```bash
pip install matplotlib networkx numpy
```

## Run Project

```bash
python main.py
```

---

# Why This Project Is Strong Academically

This project is not based on a single AI technique.

Instead, it integrates multiple AI paradigms together:

* Neural Networks
* Symbolic Reasoning
* Constraint Solving
* Search Algorithms
* Explainable AI

This makes the system a complete Hybrid AI Architecture.

---

# Future Improvements

Possible future enhancements:

* Real GPS integration
* Live traffic APIs
* Reinforcement Learning
* Deep Learning models
* Real-time sensor data
* Web dashboard
* IoT integration
* Multi-agent traffic simulation

---

# Authors

* Salman Shoaib
* Moeed Amir

---

# License

This project is developed for academic and educational purposes.
