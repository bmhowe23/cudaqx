# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np
import networkx as nx

import cudaq
from cudaq import spin
import cudaq_solvers as solvers


def test_simple_qaoa():
    Hp = 0.5 * spin.z(0) * spin.z(1) + 0.5 * spin.z(1) * spin.z(2) + \
         0.5 * spin.z(0) * spin.z(3) + 0.5 * spin.z(2) * spin.z(3)
    Href = spin.x(0) + spin.x(1) + spin.x(2) + spin.x(3)

    n_qubits = Hp.qubit_count
    n_layers = 2
    n_params = 2 * n_layers
    initial_parameters = np.random.uniform(-np.pi / 2, np.pi / 2, n_params)

    result = solvers.qaoa(Hp,
                          Href,
                          n_layers,
                          initial_parameters,
                          optimizer='cobyla')
    print(result)
    # Can unpack like a tuple
    optval, optp, config = result
    print(optval)


def test_custom_mixing_hamiltonian_execution():
    problem_ham = 0.5 * spin.z(0) * spin.z(1)
    mixing_ham = spin.x(0) + spin.x(1)
    init_params = [0.1, 0.1]

    result = solvers.qaoa(problem_ham, mixing_ham, 1, init_params)

    assert len(result.optimal_parameters) > 0
    assert len(result.optimal_parameters) == 2
    assert -1.0 <= result.optimal_value <= 1.0


def test_default_mixing_hamiltonian_execution():
    problem_ham = spin.z(0)
    init_params = [0.1, 0.1]

    result = solvers.qaoa(problem_ham, 1, init_params)

    assert len(result.optimal_parameters) > 0
    assert len(result.optimal_parameters) == 2
    assert -1.0 <= result.optimal_value <= 1.0


def test_parameter_validation():
    problem_ham = spin.z(0)
    empty_params = []

    with pytest.raises(RuntimeError):
        solvers.qaoa(problem_ham, 1, empty_params)


def test_multi_layer_execution():
    problem_ham = spin.z(0) * spin.z(1)
    init_params = [0.1, 0.1, 0.2, 0.2]  # 2 layers

    result = solvers.qaoa(problem_ham, 2, init_params)

    assert len(result.optimal_parameters) == 4
    assert -1.0 <= result.optimal_value <= 1.0


def test_overload_consistency():
    problem_ham = spin.z(0) * spin.z(1)
    mixing_ham = spin.x(0) + spin.x(1)
    init_params = [0.1, 0.1]

    result1 = solvers.qaoa(problem_ham, mixing_ham, 1, init_params)
    result2 = solvers.qaoa(problem_ham, 1, init_params)

    # Results should be similar within numerical precision
    assert abs(result1.optimal_value - result2.optimal_value) < 1e-6


def test_maxcut_single_edge():
    G = nx.Graph()
    G.add_edge(0, 1)

    ham = solvers.get_maxcut_hamiltonian(G)

    # Should have two terms: 0.5*Z0Z1 and -0.5*I0I1
    assert ham.term_count == 2
    expected_ham = 0.5 * spin.z(0) * spin.z(1) - 0.5
    assert ham == expected_ham


def test_maxcut_triangle():
    # Create triangle graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])

    ham = solvers.get_maxcut_hamiltonian(G)
    print(ham)

    # Should have 4 terms
    assert ham.term_count == 4

    # Create expected Hamiltonian using the exact structure
    expected_ham = 0.5 * spin.z(0) * spin.z(2) + 0.5 * spin.z(0) * spin.z(
        1) + 0.5 * spin.z(1) * spin.z(2) - 1.5

    # Compare Hamiltonians
    assert ham == expected_ham


def test_maxcut_disconnected():
    # Create disconnected graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])

    ham = solvers.get_maxcut_hamiltonian(G)

    # Should have 3 terms
    assert ham.term_count == 3

    # Create expected Hamiltonian using the exact structure
    expected_ham = 0.5 * spin.z(0) * spin.z(1) + 0.5 * spin.z(2) * spin.z(
        3) - 1.0

    # Compare Hamiltonians
    assert ham == expected_ham


def test_clique_single_node():
    G = nx.Graph()
    G.add_node(0, weight=1.5)

    ham = solvers.get_clique_hamiltonian(G)

    assert ham.term_count == 2
    expected_ham = 0.75 * spin.z(0) - 0.75
    assert ham == expected_ham


def test_clique_complete_graph():
    G = nx.Graph()
    node_w = {0: 2., 1: 1.5, 2: 1.}
    for node, weight in node_w.items():
        G.add_node(node, weight=weight)
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    G.add_weighted_edges_from(edges)

    ham = solvers.get_clique_hamiltonian(G, penalty=4.0)

    assert ham.term_count == 4

    expected_ham = spin.z(0) + 0.75 * spin.z(1) + 0.5 * spin.z(2) - 2.25
    assert ham == expected_ham


def test_clique_disconnected_nodes():
    G = nx.Graph()
    G.add_nodes_from([(0, {'weight': 1.0}), (1, {'weight': 1.0})])

    ham = solvers.get_clique_hamiltonian(G, penalty=2.0)

    assert ham.term_count == 2

    expected_ham = 0.5 * spin.z(0) * spin.z(1) - 0.5
    assert ham == expected_ham


def test_clique_triangle_with_disconnected():
    G = nx.Graph()
    nodes = [(i, {'weight': 1.0}) for i in range(4)]
    G.add_nodes_from(nodes)
    edges = [(0, 1), (1, 2), (0, 2)]
    G.add_edges_from(edges)

    ham = solvers.get_clique_hamiltonian(G, penalty=4.0)

    assert ham.term_count == 8

    # yapf: disable
    expected_ham  =        spin.z(2) * spin.z(3)  # IIZZ
    expected_ham +=        spin.z(1) * spin.z(3)  # IZIZ
    expected_ham +=  1.0 * spin.z(0) * spin.z(3)  # ZIIZ
    expected_ham += -2.5 * spin.z(3)              # IIIZ
    expected_ham += -0.5 * spin.z(1)              # IZII
    expected_ham +=  1.0                          # IIII
    expected_ham += -0.5 * spin.z(2)              # IIZI
    expected_ham += -0.5 * spin.z(0)              # ZIII
    # yapf: enable
    assert ham == expected_ham


def test_clique_different_penalties():
    G = nx.Graph()
    G.add_nodes_from([(0, {'weight': 1.0}), (1, {'weight': 1.0})])

    ham1 = solvers.get_clique_hamiltonian(G, penalty=2.0)
    ham2 = solvers.get_clique_hamiltonian(G, penalty=4.0)

    # Note, they *would* have the same number of terms except terms with
    # 0-valued coefficients will get trimmed, so that makes the results
    # have a different number of terms.
    assert ham1.term_count != ham2.term_count
    assert str(ham1) != str(ham2)


def test_clique_weighted_nodes():
    G = nx.Graph()
    G.add_nodes_from([(0, {'weight': 2.0}), (1, {'weight': 3.0})])
    G.add_edge(0, 1, weight=1.0)

    ham = solvers.get_clique_hamiltonian(G)

    assert ham.term_count == 3

    expected_ham = spin.z(0) + 1.5 * spin.z(1) - 2.5
    assert ham == expected_ham
