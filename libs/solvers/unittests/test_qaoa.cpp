/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq.h"
#include "cudaq/solvers/operators.h"
#include "cudaq/solvers/qaoa.h"
#include "nvqpp/test_kernels.h"

using namespace cudaq::spin;

TEST(SolversTester, checkSimpleQAOA) {

  auto Hp = 0.5 * z(0) * z(1) + 0.5 * z(1) * z(2) + 0.5 * z(0) * z(3) +
            0.5 * z(2) * z(3);
  auto Href = x(0) + x(1) + x(2) + x(3);

  const int n_qubits = Hp.num_qubits();
  const int n_layers = 2;
  const int n_params = 2 * n_layers; // * (n_qubits + Href.num_qubits());
  auto initialParameters = cudaq::random_vector(-M_PI_2, M_PI_2, n_params);

  auto optimizer = cudaq::optim::optimizer::get("cobyla");

  auto [optval, optparam, config] =
      cudaq::solvers::qaoa(Hp, Href, *optimizer, n_layers, initialParameters);
}

// Test basic QAOA execution with custom mixing Hamiltonian
TEST(QAOATest, CustomMixingHamiltonianExecution) {
  // Create a simple 2-qubit problem Hamiltonian
  cudaq::spin_op problemHam = 0.5 * cudaq::spin::z(0) * cudaq::spin::z(1);

  // Create mixing Hamiltonian (X terms)
  cudaq::spin_op mixingHam = cudaq::spin::x(0) + cudaq::spin::x(1);

  // Create optimizer
  auto opt = cudaq::optim::optimizer::get("cobyla");

  // Initial parameters for 1 layer (gamma, beta)
  std::vector<double> initParams = {0.1, 0.1};

  auto result =
      cudaq::solvers::qaoa(problemHam, mixingHam, *opt, 1, initParams);

  EXPECT_FALSE(result.optimal_parameters.empty());
  EXPECT_EQ(result.optimal_parameters.size(), 2);
  EXPECT_GE(result.optimal_value, -1.0);
  EXPECT_LE(result.optimal_value, 1.0);
}

// Test QAOA with default mixing Hamiltonian
TEST(QAOATest, DefaultMixingHamiltonianExecution) {
  // Single-qubit problem Hamiltonian
  cudaq::spin_op problemHam = cudaq::spin::z(0);

  auto opt = cudaq::optim::optimizer::get("cobyla");
  std::vector<double> initParams = {0.1, 0.1};

  auto result = cudaq::solvers::qaoa(problemHam, *opt, 1, initParams);

  EXPECT_FALSE(result.optimal_parameters.empty());
  EXPECT_EQ(result.optimal_parameters.size(), 2);
  EXPECT_GE(result.optimal_value, -1.0);
  EXPECT_LE(result.optimal_value, 1.0);
}

// Test parameter validation
TEST(QAOATest, ParameterValidation) {
  cudaq::spin_op problemHam = cudaq::spin::z(0);
  std::vector<double> emptyParams;

  EXPECT_THROW(cudaq::solvers::qaoa(problemHam, 1, emptyParams),
               std::invalid_argument);
}

// Test multi-layer QAOA
TEST(QAOATest, MultiLayerExecution) {
  cudaq::spin_op problemHam = cudaq::spin::z(0) * cudaq::spin::z(1);
  std::vector<double> initParams = {0.1, 0.1, 0.2, 0.2}; // 2 layers

  auto result = cudaq::solvers::qaoa(problemHam, 2, initParams);

  EXPECT_EQ(result.optimal_parameters.size(), 4);
  EXPECT_GE(result.optimal_value, -1.0);
  EXPECT_LE(result.optimal_value, 1.0);
}

// // Test QAOA with options
// TEST(QAOATest, OptionsHandling) {
//     cudaq::spin_op problemHam = cudaq::spin::z(0)[1];
//     std::vector<double> initParams = {0.1, 0.1};

//     cudaq::heterogeneous_map options;
//     options["shots"] = 1000;
//     options["optimizer.maxiter"] = 100;

//     auto result = cudaq::solvers::qaoa(problemHam, 1, initParams, options);

//     EXPECT_FALSE(result.optimalConfig.empty());
//     EXPECT_GE(result.optimalConfig.counts().size(), 1);
// }

// Test consistency between different QAOA overloads
TEST(QAOATest, OverloadConsistency) {
  cudaq::spin_op problemHam = cudaq::spin::z(0) * cudaq::spin::z(1);
  cudaq::spin_op mixingHam = cudaq::spin::x(0) + cudaq::spin::x(1);
  auto opt = cudaq::optim::optimizer::get("cobyla");
  std::vector<double> initParams = {0.1, 0.1};

  auto result1 =
      cudaq::solvers::qaoa(problemHam, mixingHam, *opt, 1, initParams);
  auto result2 = cudaq::solvers::qaoa(problemHam, *opt, 1, initParams);

  // Results should be similar within numerical precision
  EXPECT_NEAR(result1.optimal_value, result2.optimal_value, 1e-6);
}

TEST(MaxCutHamiltonianTest, SingleEdge) {
  cudaqx::graph g;
  g.add_edge(0, 1);

  auto ham = cudaq::solvers::get_maxcut_hamiltonian(g);
  ham.dump();
  // Should have two terms: 0.5*Z0Z1 and -0.5*I0I1
  EXPECT_EQ(ham.num_terms(), 2);

  // Verify the coefficients
  EXPECT_EQ(0.5 * cudaq::spin_op::from_word("ZZ") -
                .5 * cudaq::spin_op::from_word("II"),
            ham);
}

TEST(MaxCutHamiltonianTest, Triangle) {
  cudaqx::graph g;
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  g.add_edge(0, 2);

  auto ham = cudaq::solvers::get_maxcut_hamiltonian(g);
  ham.dump();
  // Should have 6 terms: 0.5*(Z0Z1 + Z1Z2 + Z0Z2) - 0.5*(I0I1 + I1I2 + I0I2)
  EXPECT_EQ(ham.num_terms(), 4);

  std::vector<double> data{// Term 1: ZIZ with coefficient 0.5 + 0j
                           2, 0, 2, 0.5, 0.0,

                           // Term 2: ZZI with coefficient 0.5 + 0j
                           2, 2, 0, 0.5, 0.0,

                           // Term 3: IZZ with coefficient 0.5 + 0j
                           0, 2, 2, 0.5, 0.0,

                           // Term 4: III with coefficient -1.5 + 0j
                           0, 0, 0, -1.5, 0.0,

                           // Total number of terms
                           4};
  EXPECT_EQ(ham, cudaq::spin_op(data, 3));
}

TEST(MaxCutHamiltonianTest, DisconnectedGraph) {
  cudaqx::graph g;
  g.add_edge(0, 1);
  g.add_edge(2, 3); // Disconnected edge

  auto ham = cudaq::solvers::get_maxcut_hamiltonian(g);

  // Should have 4 terms: 0.5*(Z0Z1 + Z2Z3) - 0.5*(I0I1 + I2I3)
  EXPECT_EQ(ham.num_terms(), 3);
  ham.dump();

  std::vector<double> data{// Term 1: ZZII with coefficient 0.5 + 0j
                           2, 2, 0, 0, 0.5, 0.0,

                           // Term 2: IIZZ with coefficient 0.5 + 0j
                           0, 0, 2, 2, 0.5, 0.0,

                           // Term 3: IIII with coefficient -1.0 + 0j
                           0, 0, 0, 0, -1.0, 0.0,

                           // Total number of terms
                           3};
  EXPECT_EQ(ham, cudaq::spin_op(data, 4));
}

TEST(CliqueHamiltonianTest, SingleNode) {
  cudaqx::graph g;
  g.add_node(0, 1.5); // Add node with weight 1.5

  auto ham = cudaq::solvers::get_clique_hamiltonian(g);
  ham.dump();
  EXPECT_EQ(ham.num_terms(), 2);
  EXPECT_EQ(ham, 0.75 * cudaq::spin::z(0) - .75 * cudaq::spin::i(0));
}

TEST(CliqueHamiltonianTest, CompleteGraph) {
  cudaqx::graph g;

  // Create K3 (triangle) with node and edge weights
  g.add_node(0, 2.0);
  g.add_node(1, 1.5);
  g.add_node(2, 1.0);

  g.add_edge(0, 1, 1.0);
  g.add_edge(1, 2, 1.0);
  g.add_edge(0, 2, 1.0);

  auto ham = cudaq::solvers::get_clique_hamiltonian(g, 4.0);
  ham.dump();
  EXPECT_EQ(ham.num_terms(), 4);

  std::vector<double> data{// Term 1: ZII with coefficient 1.0 + 0j
                           2, 0, 0, 1.0, 0.0,

                           // Term 2: IZI with coefficient 0.75 + 0j
                           0, 2, 0, 0.75, 0.0,

                           // Term 3: IIZ with coefficient 0.5 + 0j
                           0, 0, 2, 0.5, 0.0,

                           // Term 4: III with coefficient -2.25 + 0j
                           0, 0, 0, -2.25, 0.0,

                           // Total number of terms
                           4};
  EXPECT_EQ(ham, cudaq::spin_op(data, 3));
}

TEST(CliqueHamiltonianTest, DisconnectedNodes) {
  cudaqx::graph g;

  // Create two disconnected nodes
  g.add_node(0, 1.0);
  g.add_node(1, 1.0);

  auto ham = cudaq::solvers::get_clique_hamiltonian(g, 2.0);
  ham.dump();
  // Should have 2 vertex terms + 1 penalty term for the non-edge
  EXPECT_EQ(ham.num_terms(), 4);
  std::vector<double> data{// Term 1: ZZ with coefficient 0.5 + 0j
                           2, 2, 0.5, 0.0,

                           // Term 2: ZI with coefficient 0.0 + 0j
                           2, 0, 0.0, 0.0,

                           // Term 3: IZ with coefficient 0.0 + 0j
                           0, 2, 0.0, 0.0,

                           // Term 4: II with coefficient -0.5 + 0j
                           0, 0, -0.5, 0.0,

                           // Total number of terms
                           4};
  EXPECT_EQ(ham, cudaq::spin_op(data, 2));
}

TEST(CliqueHamiltonianTest, TriangleWithDisconnectedNode) {
  cudaqx::graph g;

  // Create K3 + one disconnected node
  g.add_node(0, 1.0);
  g.add_node(1, 1.0);
  g.add_node(2, 1.0);
  g.add_node(3, 1.0);

  g.add_edge(0, 1, 1.0);
  g.add_edge(1, 2, 1.0);
  g.add_edge(0, 2, 1.0);

  auto none_edges = g.get_disconnected_vertices();
  for (auto &ee : none_edges)
    printf("%d %d \n", ee.first, ee.second);
  auto ham = cudaq::solvers::get_clique_hamiltonian(g, 4.0);
  ham.dump();

  EXPECT_EQ(ham.num_terms(), 8);

  std::vector<double> data{// Term 1: IIZZ with coefficient 1.0 + 0j
                           0, 0, 2, 2, 1.0, 0.0,

                           // Term 2: IZIZ with coefficient 1.0 + 0j
                           0, 2, 0, 2, 1.0, 0.0,

                           // Term 3: ZIIZ with coefficient 1.0 + 0j
                           2, 0, 0, 2, 1.0, 0.0,

                           // Term 4: IIIZ with coefficient -2.5 + 0j
                           0, 0, 0, 2, -2.5, 0.0,

                           // Term 5: IZII with coefficient -0.5 + 0j
                           0, 2, 0, 0, -0.5, 0.0,

                           // Term 6: IIII with coefficient 1.0 + 0j
                           0, 0, 0, 0, 1.0, 0.0,

                           // Term 7: IIZI with coefficient -0.5 + 0j
                           0, 0, 2, 0, -0.5, 0.0,

                           // Term 8: ZIII with coefficient -0.5 + 0j
                           2, 0, 0, 0, -0.5, 0.0,

                           // Total number of terms
                           8};
  EXPECT_EQ(ham, cudaq::spin_op(data, 4));
}

TEST(CliqueHamiltonianTest, DifferentPenalties) {
  cudaqx::graph g;

  // Create two disconnected nodes
  g.add_node(0, 1.0);
  g.add_node(1, 1.0);

  auto ham1 = cudaq::solvers::get_clique_hamiltonian(g, 2.0);
  auto ham2 = cudaq::solvers::get_clique_hamiltonian(g, 4.0);

  // Same number of terms but different coefficients
  EXPECT_EQ(ham1.num_terms(), ham2.num_terms());
  EXPECT_NE(ham1.to_string(), ham2.to_string());
}

TEST(CliqueHamiltonianTest, WeightedNodes) {
  cudaqx::graph g;

  // Create two connected nodes with different weights
  g.add_node(0, 2.0);
  g.add_node(1, 3.0);
  g.add_edge(0, 1, 1.0);

  auto ham = cudaq::solvers::get_clique_hamiltonian(g);
  ham.dump();
  // Should have 2 vertex terms with different coefficients
  EXPECT_EQ(ham.num_terms(), 3);
  std::vector<double> data{// Term 1: ZI with coefficient 1.0 + 0j
                           2, 0, 1.0, 0.0,

                           // Term 2: IZ with coefficient 1.5 + 0j
                           0, 2, 1.5, 0.0,

                           // Term 3: II with coefficient -2.5 + 0j
                           0, 0, -2.5, 0.0,

                           // Total number of terms
                           3};
  EXPECT_EQ(ham, cudaq::spin_op(data, 2));
}

TEST(QAOAMaxCutTest, SingleEdge) {
  // Create simple graph with single edge
  cudaqx::graph g;
  g.add_edge(0, 1);

  // Get MaxCut Hamiltonian
  auto ham = cudaq::solvers::get_maxcut_hamiltonian(g);

  // Set up QAOA parameters
  std::size_t num_layers = 1;
  std::vector<double> initial_params = {0.5, 0.5}; // gamma, beta

  // Run QAOA
  auto result = cudaq::solvers::qaoa(ham, num_layers, initial_params,
                                     {{"verbose", true}});

  // Verify results
  EXPECT_GT(std::abs(result.optimal_value),
            0.5); // Should be better than random guess
  EXPECT_EQ(result.optimal_parameters.size(), 2 * num_layers);
}

TEST(QAOAMaxCutTest, Triangle) {
  cudaqx::graph g;
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  g.add_edge(0, 2);

  auto ham = cudaq::solvers::get_maxcut_hamiltonian(g);
  ham.dump();
  // Try with 2 QAOA layers
  std::size_t num_layers = 2;
  std::vector<double> initial_params = {0.5, 0.5, 0.5,
                                        0.5}; // gamma1, beta1, gamma2, beta2

  auto result = cudaq::solvers::qaoa(ham, num_layers, initial_params,
                                     {{"verbose", true}});

  result.optimal_config.dump();

  // For triangle, max cut value should be 2
  EXPECT_NEAR(std::abs(result.optimal_value), 2.0, 0.1);
}

TEST(QAOAMaxCutTest, WeightedGraph) {
  cudaqx::graph g;
  g.add_edge(0, 1, 2.0);
  g.add_edge(1, 2, 1.0);
  g.add_edge(0, 2, 0.5);

  auto ham = cudaq::solvers::get_maxcut_hamiltonian(g);

  std::size_t num_layers = 3;
  std::vector<double> initial_params(2 * num_layers, 0.5);

  auto result = cudaq::solvers::qaoa(ham, num_layers, initial_params);

  // Max weighted cut should be at least 2.5
  EXPECT_GT(std::abs(result.optimal_value), 2.4);
}

TEST(QAOAMaxCutTest, CustomMixer) {
  cudaqx::graph g;
  g.add_edge(0, 1);
  g.add_edge(1, 2);

  auto problem_ham = cudaq::solvers::get_maxcut_hamiltonian(g);

  // Create custom X-mixer Hamiltonian
  auto mixer_ham = cudaq::spin::x(0) + cudaq::spin::x(1) + cudaq::spin::x(2);

  std::size_t num_layers = 2;
  std::vector<double> initial_params = {0.5, 0.5, 0.5, 0.5};

  auto result =
      cudaq::solvers::qaoa(problem_ham, mixer_ham, num_layers, initial_params);

  EXPECT_GT(std::abs(result.optimal_value), 1.0);
}

TEST(QAOAMaxCutTest, DisconnectedGraph) {
  cudaqx::graph g;
  g.add_edge(0, 1);
  g.add_edge(2, 3); // Disconnected component

  auto ham = cudaq::solvers::get_maxcut_hamiltonian(g);

  std::size_t num_layers = 2;
  std::vector<double> initial_params(2 * num_layers, 0.5);

  auto result = cudaq::solvers::qaoa(ham, num_layers, initial_params);

  // Should find max cut of 2 (one cut per component)
  EXPECT_NEAR(std::abs(result.optimal_value), 2.0, 0.1);

  // Check measurement results
  // auto counts = result.optimal_config;
  // EXPECT_GT(counts.size(), 0);
}

TEST(QAOAMaxCutTest, ParameterOptimization) {
  cudaqx::graph g;
  g.add_edge(0, 1);
  g.add_edge(1, 2);

  auto ham = cudaq::solvers::get_maxcut_hamiltonian(g);

  // Try different initial parameters
  std::vector<double> params1 = {0.1, 0.1};
  std::vector<double> params2 = {1.0, 1.0};

  auto result1 = cudaq::solvers::qaoa(ham, 1, params1);
  auto result2 = cudaq::solvers::qaoa(ham, 1, params2);

  // Both should converge to similar optimal values
  EXPECT_NEAR(std::abs(result1.optimal_value), std::abs(result2.optimal_value),
              0.1);
}