# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import numpy as np
import cudaq_qec as qec


def create_test_matrix():
    np.random.seed(42)
    return np.random.randint(0, 2, (10, 20)).astype(np.uint8)


def create_test_syndrome():
    np.random.seed(42)
    return np.random.random(10).tolist()


H = create_test_matrix()


def test_decoder_initialization():
    decoder = qec.get_decoder('example_byod', H)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_initialization_with_error():
    # We do not support column-major order (Fortran order)
    H_bad = np.zeros((10, 20), dtype=np.uint8, order='F')
    with pytest.raises(RuntimeError) as e:
        decoder = qec.get_decoder('single_error_lut_example', H_bad)


def test_decoder_api():
    # Test decode_batch
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode_batch(
        [create_test_syndrome(), create_test_syndrome()])
    assert len(result) == 2
    for r in result:
        assert hasattr(r, 'converged')
        assert hasattr(r, 'result')
        assert isinstance(r.converged, bool)
        assert isinstance(r.result, list)
        assert len(r.result) == 10

    # Test decode_async
    decoder = qec.get_decoder('example_byod', H)
    result_async = decoder.decode_async(create_test_syndrome())
    assert hasattr(result_async, 'get')
    assert hasattr(result_async, 'ready')

    result = result_async.get()
    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, list)
    assert len(result.result) == 10


def test_decoder_result_structure():
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode(create_test_syndrome())

    # Test basic structure
    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert hasattr(result, 'opt_results')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, list)
    assert len(result.result) == 10

    # Test opt_results functionality
    assert result.opt_results is None  # Default should be None

    # Test that opt_results is preserved in async decode
    async_result = decoder.decode_async(create_test_syndrome())
    result = async_result.get()
    assert hasattr(result, 'opt_results')
    assert result.opt_results is None


def test_decoder_plugin_initialization():
    decoder = qec.get_decoder('single_error_lut_example', H)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_double_vec():
    vec = np.array([1, 2, 3], dtype=np.float64)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_float_vec():
    vec = np.array([1, 2, 3], dtype=np.float32)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_uint8_vec():
    vec = np.array([1, 2, 3], dtype=np.uint8)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_int32_vec():
    vec = np.array([1, 2, 3], dtype=np.int32)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_int16_vec():
    vec = np.array([1, 2, 3], dtype=np.int16)
    with pytest.raises(RuntimeError) as e:
        decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert "Unsupported array data type" in repr(e)


def test_decoder_plugin_result_structure():
    decoder = qec.get_decoder('single_error_lut_example', H)
    result = decoder.decode(create_test_syndrome())

    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, list)


def test_decoder_result_values():
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode(create_test_syndrome())

    assert result.converged is True
    assert all(isinstance(x, float) for x in result.result)
    assert all(0 <= x <= 1 for x in result.result)


@pytest.mark.parametrize("matrix_shape,syndrome_size", [((5, 10), 5),
                                                        ((15, 30), 15),
                                                        ((20, 40), 20)])
def test_decoder_different_matrix_sizes(matrix_shape, syndrome_size):
    np.random.seed(42)
    H = np.random.randint(0, 2, matrix_shape).astype(np.uint8)
    syndrome = np.random.random(syndrome_size).tolist()

    decoder = qec.get_decoder('example_byod', H)
    convergence, result, opt = decoder.decode(syndrome)

    assert len(result) == syndrome_size
    assert convergence is True
    assert all(isinstance(x, float) for x in result)
    assert all(0 <= x <= 1 for x in result)


# FIXME add this back
# def test_decoder_error_handling():
#     H = Tensor(create_test_matrix())
#     decoder = qec.get_decoder('example_byod', H)

#     # Test with incorrect syndrome size
#     with pytest.raises(ValueError):
#         wrong_syndrome = np.random.random(15).tolist()  # Wrong size
#         decoder.decode(wrong_syndrome)

#     # Test with invalid syndrome type
#     with pytest.raises(TypeError):
#         wrong_type_syndrome = "invalid"
#         decoder.decode(wrong_type_syndrome)


def test_decoder_reproducibility():
    decoder = qec.get_decoder('example_byod', H)

    np.random.seed(42)
    convergence1, result1, opt1 = decoder.decode(create_test_syndrome())

    np.random.seed(42)
    convergence2, result2, opt2 = decoder.decode(create_test_syndrome())

    assert result1 == result2
    assert convergence1 == convergence2


def test_pass_weights():
    error_probability = 0.1
    weights = np.ones(H.shape[1]) * np.log(
        (1 - error_probability) / error_probability)
    decoder = qec.get_decoder('example_byod', H, weights=weights)
    # Test is that no error is thrown


def test_sort_pcm_columns_non_decreasing_column_weight():
    # Create a test parity-check matrix with random binary values.
    # yapf: disable
    H = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 1],
                  [1, 0, 0, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 1]],
                 dtype=np.uint8)
    # yapf: enable

    H_calculated = qec.sort_pcm_columns(H)

    # yapf: disable
    H_expected = np.array(
        [[1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 1, 0],
         [0, 1, 1, 0, 0, 0, 1, 1, 1]],
        dtype=np.uint8)
    # yapf: enable

    assert np.array_equal(H_calculated, H_expected)

    col_order = qec.get_sorted_pcm_column_indices(H)
    expected_order = [1, 8, 4, 0, 3, 2, 6, 7, 5]
    assert col_order == expected_order

    # Now check that reordering the columns of H yields H_expected
    H_reordered = qec.reorder_pcm_columns(H, col_order)
    assert np.array_equal(H_reordered, H_expected)


def test_sort_pcm_columns_invalid_input():
    # Test that passing a non-2D array raises an error.
    H_invalid = np.array([0, 1, 0, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError):
        qec.get_sorted_pcm_column_indices(H_invalid)


def test_gen_random_pcm():
    pcm = qec.generate_random_pcm(n_rounds=10,
                                  n_errs_per_round=20,
                                  n_syndromes_per_round=10,
                                  weight=3,
                                  seed=13)
    is_sorted = qec.pcm_is_sorted(pcm)
    assert is_sorted is False
    pcm = qec.sort_pcm_columns(pcm)
    is_sorted = qec.pcm_is_sorted(pcm)
    assert is_sorted is True
    print('')
    qec.dump_pcm(pcm)
    print('')
    assert pcm.shape == (100, 200)


def test_get_pcm_for_rounds():
    pcm = qec.generate_random_pcm(n_rounds=10,
                                  n_errs_per_round=20,
                                  n_syndromes_per_round=10,
                                  weight=3,
                                  seed=13)
    pcm = qec.sort_pcm_columns(pcm)
    pcm_for_rounds = qec.get_pcm_for_rounds(pcm, 10, 0, 1)
    assert pcm_for_rounds.shape == (20, 30)
    print('')
    qec.dump_pcm(pcm_for_rounds)
    print('')


def test_shuffle_pcm_columns():
    pcm = qec.generate_random_pcm(n_rounds=10,
                                  n_errs_per_round=20,
                                  n_syndromes_per_round=10,
                                  weight=3,
                                  seed=13)
    sorted_pcm = qec.sort_pcm_columns(pcm)
    shuffled_pcm = qec.shuffle_pcm_columns(sorted_pcm, seed=13)

    # They should not be equal here
    assert not np.array_equal(sorted_pcm, shuffled_pcm)

    # They should be equal after sorting
    assert np.array_equal(qec.sort_pcm_columns(shuffled_pcm), sorted_pcm)


def test_simplify_pcm():
    syndromes_per_round = 10
    pcm = qec.generate_random_pcm(
        n_rounds=10,
        n_errs_per_round=30,
        n_syndromes_per_round=syndromes_per_round,
        weight=1,  # force some duplicate columns for this test
        seed=13)
    weights = np.ones(pcm.shape[1]) * 0.01
    new_pcm, new_weights = qec.simplify_pcm(pcm, weights, syndromes_per_round)
    # qec.dump_pcm(new_pcm)
    print(new_pcm.shape)
    assert new_pcm.shape[0] == pcm.shape[0]
    assert new_pcm.shape[1] < pcm.shape[1]  # we expect fewer columns
    assert new_weights.shape == (new_pcm.shape[1],)

    # Test that the new weights are not all uniform.
    assert not np.allclose(new_weights, new_weights[0])


def test_version():
    decoder = qec.get_decoder('example_byod', H)
    assert "CUDA-Q QEC Base Decoder" in decoder.get_version()


def test_single_error_lut_opt_results():
    # Test with invalid opt_results
    invalid_args = {"opt_results": {"invalid_type": True}}
    with pytest.raises(RuntimeError) as e:
        decoder = qec.get_decoder("single_error_lut", H, **invalid_args)
        decoder.decode(create_test_syndrome())
    assert "Requested result types not available" in str(e.value)

    # Test with valid opt_results
    valid_args = {
        "opt_results": {
            "error_probability": True,
            "syndrome_weight": True,
            "decoding_time": False,
            "num_repetitions": 5
        }
    }
    decoder = qec.get_decoder("single_error_lut", H, **valid_args)
    result = decoder.decode(create_test_syndrome())

    # Verify opt_results
    assert result.opt_results is not None
    assert "error_probability" in result.opt_results
    assert "syndrome_weight" in result.opt_results
    assert "decoding_time" not in result.opt_results  # Was set to False
    assert "num_repetitions" in result.opt_results
    assert result.opt_results["num_repetitions"] == 5


if __name__ == "__main__":
    pytest.main()
