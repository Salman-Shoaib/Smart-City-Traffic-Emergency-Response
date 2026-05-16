"""
=============================================================================
Module: ann_module.py
Description:
    ANN Priority Prediction Module for Smart City Traffic & Emergency
    Response AI System.
    Implements a simple Multi-Layer Perceptron (MLP) from scratch using
    only Python and math - no external ML libraries required.
    The network takes 5 input features and outputs a priority class:
        0 = Normal, 1 = High, 2 = Critical
=============================================================================
"""

import math
import random


def sigmoid(x: float) -> float:
    """
    -------------------------------------------------------------------------
    Function: sigmoid
    Description:
        Sigmoid activation function. Squashes input to range (0, 1).
    -------------------------------------------------------------------------
    """
    clipped = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-clipped))


def softmax(values: list) -> list:
    """
    -------------------------------------------------------------------------
    Function: softmax
    Description:
        Softmax activation for output layer.
    -------------------------------------------------------------------------
    """
    max_value = max(values)
    shifted_values = [math.exp(value - max_value) for value in values]
    total = sum(shifted_values)
    return [value / total for value in shifted_values]


def _one_hot(size: int, index: int) -> list:
    """Build a one-hot encoded target vector."""
    values = [0.0] * size
    values[index] = 1.0
    return values


def _argmax(values: list) -> int:
    """Return the index of the largest value."""
    return values.index(max(values))


class SimpleMLP:
    """
    -------------------------------------------------------------------------
    Class: SimpleMLP
    Description:
        A simple 3-layer MLP: Input(5) -> Hidden(8) -> Hidden(6) -> Output(3)
        Trained using backpropagation with gradient descent.
    -------------------------------------------------------------------------
    """

    def __init__(self, input_size=5, hidden1=8, hidden2=6, output_size=3, lr=0.1):
        self.lr = lr
        random.seed(42)
        self.W1 = self._random_matrix(hidden1, input_size)
        self.B1 = self._random_vector(hidden1)
        self.W2 = self._random_matrix(hidden2, hidden1)
        self.B2 = self._random_vector(hidden2)
        self.W3 = self._random_matrix(output_size, hidden2)
        self.B3 = self._random_vector(output_size)

    def _random_matrix(self, rows: int, cols: int) -> list:
        """Create a matrix filled with random values."""
        return [[random.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)]

    def _random_vector(self, size: int) -> list:
        """Create a vector filled with random values."""
        return [random.uniform(-0.5, 0.5) for _ in range(size)]

    def _normalize_features(self, features: list) -> list:
        """Normalize the five ANN input features to small numeric ranges."""
        divisors = [3.0, 3.0, 2.0, 2.0, 10.0]
        return [value / divisor for value, divisor in zip(features, divisors)]

    def _forward_layer(self, inputs: list, weights: list, biases: list, activation=sigmoid) -> list:
        """Run one dense layer forward pass."""
        outputs = []
        for weight_row, bias in zip(weights, biases):
            total = sum(weight * value for weight, value in zip(weight_row, inputs)) + bias
            outputs.append(activation(total) if activation else total)
        return outputs

    def _output_scores(self, hidden_values: list) -> list:
        """Calculate raw output scores before softmax."""
        return [
            sum(weight * value for weight, value in zip(weight_row, hidden_values)) + bias
            for weight_row, bias in zip(self.W3, self.B3)
        ]

    def _hidden_delta(self, next_delta: list, next_weights: list, activations: list) -> list:
        """Backpropagate error into a hidden layer."""
        deltas = []
        for index, activation in enumerate(activations):
            propagated_error = sum(delta * weights[index] for delta, weights in zip(next_delta, next_weights))
            deltas.append(propagated_error * activation * (1 - activation))
        return deltas

    def _update_layer(self, weights: list, biases: list, delta: list, inputs: list):
        """Apply gradient-descent update to one layer."""
        for row_index, row_delta in enumerate(delta):
            for col_index, input_value in enumerate(inputs):
                weights[row_index][col_index] -= self.lr * row_delta * input_value
            biases[row_index] -= self.lr * row_delta

    def forward(self, features: list) -> tuple:
        """
        ---------------------------------------------------------------------
        Function: forward
        Description:
            Forward pass through all 3 layers.
        ---------------------------------------------------------------------
        """
        normalized = self._normalize_features(features)
        hidden1 = self._forward_layer(normalized, self.W1, self.B1, sigmoid)
        hidden2 = self._forward_layer(hidden1, self.W2, self.B2, sigmoid)
        outputs = softmax(self._output_scores(hidden2))
        return normalized, hidden1, hidden2, outputs

    def predict(self, features: list) -> int:
        """Return the predicted class index."""
        _, _, _, outputs = self.forward(features)
        return _argmax(outputs)

    def predict_with_confidence(self, features: list) -> tuple:
        """Return (class_index, confidence_percent, all_probabilities)."""
        _, _, _, outputs = self.forward(features)
        predicted_class = _argmax(outputs)
        confidence = round(outputs[predicted_class] * 100, 2)
        return predicted_class, confidence, outputs

    def _backprop(self, features: list, true_label: int):
        """Run one backpropagation step and update the network."""
        normalized, hidden1, hidden2, outputs = self.forward(features)
        expected = _one_hot(len(outputs), true_label)
        output_delta = [output - target for output, target in zip(outputs, expected)]
        hidden2_delta = self._hidden_delta(output_delta, self.W3, hidden2)
        hidden1_delta = self._hidden_delta(hidden2_delta, self.W2, hidden1)

        self._update_layer(self.W3, self.B3, output_delta, hidden2)
        self._update_layer(self.W2, self.B2, hidden2_delta, hidden1)
        self._update_layer(self.W1, self.B1, hidden1_delta, normalized)

    def train(self, dataset: list, epochs: int = 500):
        """
        ---------------------------------------------------------------------
        Function: train
        Description:
            Trains the MLP on a list of (features, label) pairs.
        ---------------------------------------------------------------------
        """
        for _ in range(epochs):
            random.shuffle(dataset)
            for features, label in dataset:
                self._backprop(features, label)


TRAINING_DATA = [
    ([3, 0, 0, 0, 3], 0),
    ([3, 0, 0, 1, 4], 0),
    ([3, 1, 0, 0, 5], 0),
    ([3, 0, 1, 0, 3], 0),
    ([3, 1, 1, 1, 4], 0),
    ([3, 0, 0, 2, 6], 0),
    ([0, 1, 1, 1, 5], 1),
    ([1, 1, 1, 1, 4], 1),
    ([2, 1, 1, 1, 3], 1),
    ([0, 2, 1, 1, 6], 1),
    ([1, 2, 0, 1, 5], 1),
    ([2, 1, 2, 0, 4], 1),
    ([0, 1, 2, 1, 7], 1),
    ([0, 2, 2, 2, 8], 2),
    ([0, 3, 2, 2, 9], 2),
    ([1, 2, 2, 2, 6], 2),
    ([0, 2, 2, 1, 5], 2),
    ([1, 3, 2, 2, 7], 2),
    ([2, 2, 2, 2, 5], 2),
    ([0, 3, 2, 2, 4], 2),
    ([1, 2, 1, 2, 8], 2),
]

PRIORITY_LABELS = {0: "Normal", 1: "High", 2: "Critical"}

_model = None


def get_trained_model() -> SimpleMLP:
    """
    -------------------------------------------------------------------------
    Function: get_trained_model
    Description:
        Returns a singleton trained ANN model. Trains on first call.
    -------------------------------------------------------------------------
    """
    global _model
    if _model is None:
        _model = SimpleMLP(lr=0.15)
        _model.train(TRAINING_DATA, epochs=1000)
    return _model


def predict_priority(ann_features: list) -> dict:
    """
    -------------------------------------------------------------------------
    Function: predict_priority
    Description:
        Main entry point for the ANN module.
    -------------------------------------------------------------------------
    """
    predicted_class, confidence, probabilities = get_trained_model().predict_with_confidence(ann_features)
    return {
        "priority_class": predicted_class,
        "priority_label": PRIORITY_LABELS[predicted_class],
        "confidence": confidence,
        "probabilities": {
            "Normal": round(probabilities[0] * 100, 2),
            "High": round(probabilities[1] * 100, 2),
            "Critical": round(probabilities[2] * 100, 2),
        },
    }
