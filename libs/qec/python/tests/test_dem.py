# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import numpy as np
import cudaq
import cudaq_qec as qec


# Helper function to convert a binary matrix to a convenient string
def mat_to_str(mat):
    s = ''
    if mat.ndim == 1:
        for col in mat:
            if col == 1:
                s += '1'
            else:
                s += '.'
        s += '\n'
        return s
    else:
        for row in mat:
            for col in row:
                if col == 1:
                    s += '1'
                else:
                    s += '.'
            s += '\n'
    return s


# Use the fixture to set the target
@pytest.fixture(scope="module", autouse=True)
def set_target():
    cudaq.set_target("stim")
    yield
    cudaq.reset_target()


def test_dem_from_memory_circuit():
    code = qec.get_code('steane')
    # observables = code.get_pauli_observables_matrix()
    p = 0.01
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 2

    dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)
    expected_detector_error_matrix = """
1111...1..............
.1.111..111...........
..11.11..1.1111.......
.......111.1.1.1111...
..........1.11..1.111.
..............1..11.11
"""
    # print(mat_to_str(dem.detector_error_matrix), end='')
    assert '\n' + mat_to_str(
        dem.detector_error_matrix) == expected_detector_error_matrix

    # Round to 4 decimal places
    # print(np.round(dem.error_rates, 4))
    expected_error_rates = [
        0.0183, 0.0235, 0.0158, 0.0209, 0.0310, 0.0235, 0.0183, 0.0106, 0.0053,
        0.0053, 0.0106, 0.0053, 0.0053, 0.0053, 0.0106, 0.0235, 0.0158, 0.0158,
        0.0183, 0.0335, 0.0209, 0.0434
    ]
    assert np.allclose(dem.error_rates, expected_error_rates, atol=1e-4)

    expected_observables_flips_matrix = '1....11.....1......111\n'
    # print(mat_to_str(dem.observables_flips_matrix), end='')
    assert mat_to_str(
        dem.observables_flips_matrix) == expected_observables_flips_matrix


def test_decoding_from_dem_from_memory_circuit():
    cudaq.set_random_seed(13)
    code = qec.get_code('steane')
    observables = code.get_pauli_observables_matrix()
    Lz = code.get_observables_z()
    p = 0.001
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 6
    nShots = 200

    dem = qec.dem_from_memory_circuit(code, statePrep, nRounds, noise)
    # Sample the memory circuit with errors
    syndromes, data = qec.sample_memory_circuit(code, statePrep, nShots,
                                                nRounds, noise)

    logical_measurements = (Lz @ data.transpose()) % 2
    # only one logical qubit, so do not need the second axis
    logical_measurements = logical_measurements.flatten()

    # Use nShots for the first dimension and whatever is left for the second
    syndromes = syndromes.reshape((nShots, -1))
    decoder = qec.get_decoder('single_error_lut', dem.detector_error_matrix)
    dr = decoder.decode_batch(syndromes)
    error_predictions = np.array([e.result for e in dr], dtype=np.uint8)
    dr_converged = np.array([e.converged for e in dr], dtype=np.uint8)

    data_predictions = (dem.observables_flips_matrix @ error_predictions.T) % 2
    print(f'data_predictions.shape: {data_predictions.shape}')
    print(f'dr_converged        : {mat_to_str(dr_converged)}', end='')
    print(f'data_predictions    : {mat_to_str(data_predictions)}', end='')
    print(f'logical_measurements: {mat_to_str(logical_measurements)}', end='')

    nLogicalErrorsWithoutDecoding = np.sum(logical_measurements)
    nLogicalErrorsWithDecoding = np.sum(data_predictions ^ logical_measurements)
    print(f'nLogicalErrorsWithoutDecoding : {nLogicalErrorsWithoutDecoding}')
    print(f'nLogicalErrorsWithDecoding    : {nLogicalErrorsWithDecoding}')
    assert nLogicalErrorsWithDecoding < nLogicalErrorsWithoutDecoding


if __name__ == "__main__":
    pytest.main()
