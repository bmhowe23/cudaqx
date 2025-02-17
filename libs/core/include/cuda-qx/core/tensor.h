/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "extension_point.h"
#include "tensor_impl.h"
#include "type_traits.h"

namespace cudaqx {

/// @brief A tensor class implementing the PIMPL idiom. The flattened data is
/// stored where the strides grow from right to left (similar to a
/// multi-dimensional C array).
template <typename Scalar = std::complex<double>>
class tensor {
private:
  std::shared_ptr<details::tensor_impl<Scalar>> pimpl;

  tensor<Scalar> mv_product(const tensor<Scalar> &vec) const {
    if (rank() != 2 || vec.rank() != 1) {
      throw std::runtime_error(
          "Matrix-vector product requires rank-2 matrix and rank-1 vector");
    }
    if (shape()[1] != vec.shape()[0]) {
      throw std::runtime_error("Invalid dimensions for matrix-vector product");
    }

    tensor<Scalar> result({shape()[0]});
    pimpl->matrix_vector_product(vec.pimpl.get(), result.pimpl.get());
    return result;
  }

public:
  /// @brief Type alias for the scalar type used in the tensor
  using scalar_type = typename details::tensor_impl<Scalar>::scalar_type;
  static constexpr auto ScalarAsString = type_to_string<Scalar>();

  /// @brief Construct an empty tensor
  tensor()
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(std::string("xtensor") +
                                                  std::string(ScalarAsString),
                                              std::vector<std::size_t>())
                .release())) {}

  /// @brief Construct a tensor with the given shape
  /// @param shape The shape of the tensor
  tensor(const std::vector<std::size_t> &shape)
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(
                std::string("xtensor") + std::string(ScalarAsString), shape)
                .release())) {}

  /// @brief Construct a tensor with the given data and shape
  /// @param data Pointer to the tensor data. This takes ownership of the data.
  /// @param shape The shape of the tensor
  tensor(const scalar_type *data, const std::vector<std::size_t> &shape)
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(std::string("xtensor") +
                                                  std::string(ScalarAsString),
                                              data, shape)
                .release())) {}

  /// @brief Construct a tensor with the given bitstrings
  /// @param data Bitstrings from which to construct tensor
  template <typename T, typename = std::enable_if_t<
                            std::is_convertible<T, std::string>::value>>
  tensor(const std::vector<T> &data)
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(
                std::string("xtensor") + std::string(ScalarAsString), data)
                .release())) {}

  /// @brief Get the rank of the tensor
  /// @return The rank of the tensor
  std::size_t rank() const { return pimpl->rank(); }

  /// @brief Get the total size of the tensor
  /// @return The total number of elements in the tensor
  std::size_t size() const { return pimpl->size(); }

  /// @brief Get the shape of the tensor
  /// @return A vector containing the dimensions of the tensor
  std::vector<std::size_t> shape() const { return pimpl->shape(); }

  /// @brief Access a mutable element of the tensor
  /// @param indices The indices of the element to access
  /// @return A reference to the element at the specified indices
  scalar_type &at(const std::vector<size_t> &indices) {
    if (indices.size() != rank())
      throw std::runtime_error("Invalid indices provided to tensor::at(), size "
                               "must be equal to rank.");
    return pimpl->at(indices);
  }

  /// @brief Access a const element of the tensor
  /// @param indices The indices of the element to access
  /// @return A const reference to the element at the specified indices
  const scalar_type &at(const std::vector<size_t> &indices) const {
    return pimpl->at(indices);
  }

  /// @brief Copy data into the tensor
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  void copy(const scalar_type *data,
            const std::vector<std::size_t> shape = {}) {
    if (pimpl->shape().empty() && shape.empty())
      throw std::runtime_error(
          "This tensor does not have a shape yet, must provide one to copy()");

    pimpl->copy(data, pimpl->shape().empty() ? shape : pimpl->shape());
  }

  /// @brief Take ownership of the given data
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  void take(const scalar_type *data,
            const std::vector<std::size_t> shape = {}) {
    if (pimpl->shape().empty() && shape.empty())
      throw std::runtime_error(
          "This tensor does not have a shape yet, must provide one to take()");

    pimpl->take(data, pimpl->shape().empty() ? shape : pimpl->shape());
  }

  /// @brief Borrow the given data without taking ownership
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  void borrow(const scalar_type *data,
              const std::vector<std::size_t> shape = {}) {
    if (pimpl->shape().empty() && shape.empty())
      throw std::runtime_error("This tensor does not have a shape yet, must "
                               "provide one to borrow()");

    pimpl->borrow(data, pimpl->shape().empty() ? shape : pimpl->shape());
  }

  // Scalar-resulting operations
  Scalar sum_all() const { return pimpl->sum_all(); }

  // Boolean-resulting operations
  bool any() const { return pimpl->any(); }

  // Elementwise operations
  tensor<Scalar> operator+(const tensor<Scalar> &other) const {
    if (shape() != other.shape()) {
      throw std::runtime_error("Tensor shapes must match for addition");
    }
    tensor<Scalar> result(shape());
    pimpl->elementwise_add(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  tensor<Scalar> operator*(const tensor<Scalar> &other) const {
    if (shape() != other.shape()) {
      throw std::runtime_error("Tensor shapes must match for multiplication");
    }
    tensor<Scalar> result(shape());
    pimpl->elementwise_multiply(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  tensor<Scalar> operator%(const tensor<Scalar> &other) const {
    if (shape() != other.shape()) {
      throw std::runtime_error("Tensor shapes must match for modulo");
    }
    tensor<Scalar> result(shape());
    pimpl->elementwise_modulo(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  // Tensor-Scalar operations
  tensor<Scalar> operator%(Scalar value) const {
    tensor<Scalar> result(shape());
    pimpl->scalar_modulo(value, result.pimpl.get());
    return result;
  }

  tensor<Scalar> slice(const std::vector<std::size_t> &v_begin,
                       const std::vector<std::size_t> &v_end) {
    Scalar __restrict__ *dst = nullptr;
    std::size_t new_size = 0;
    std::vector<std::size_t> new_shape(rank());
    for (std::size_t r = 0; r < rank(); r++) {
      new_shape[r] = v_end[r] - v_begin[r] + 1;
      if (r == 0)
        new_size = new_shape[r];
      else
        new_size *= new_shape[r];
    }
    dst = new Scalar[new_size];
    size_t ix = 0;
    if (rank() == 1) {
      const Scalar __restrict__ *src = data();
      for (size_t a = v_begin[0]; a <= v_end[0]; a++)
        dst[ix++] = src[a];
    } else if (rank() == 2) {
      for (size_t a = v_begin[0]; a <= v_end[0]; a++) {
        const Scalar __restrict__ *src = &at({a, 0});
        for (size_t b = v_begin[1]; b <= v_end[1]; b++)
          dst[ix++] = src[b];
      }
    } else if (rank() == 3) {
      for (size_t a = v_begin[0]; a <= v_end[0]; a++)
        for (size_t b = v_begin[1]; b <= v_end[1]; b++) {
          const Scalar __restrict__ *src = &at({a, b, 0});
          for (size_t c = v_begin[2]; c <= v_end[2]; c++)
            dst[ix++] = src[c];
        }
    } else if (rank() == 4) {
      for (size_t a = v_begin[0]; a <= v_end[0]; a++)
        for (size_t b = v_begin[1]; b <= v_end[1]; b++)
          for (size_t c = v_begin[2]; c <= v_end[2]; c++) {
            const Scalar __restrict__ *src = &at({a, b, c, 0});
            for (size_t d = v_begin[3]; d <= v_end[3]; d++)
              dst[ix++] = src[d];
          }
    } else if (rank() == 5) {
      for (size_t a = v_begin[0]; a <= v_end[0]; a++)
        for (size_t b = v_begin[1]; b <= v_end[1]; b++)
          for (size_t c = v_begin[2]; c <= v_end[2]; c++)
            for (size_t d = v_begin[3]; d <= v_end[3]; d++) {
              const Scalar __restrict__ *src = &at({a, b, c, d, 0});
              for (size_t e = v_begin[4]; e <= v_end[4]; e++)
                dst[ix++] = src[e];
            }
    } else {
      // Throw error (TODO)
    }
    return cudaqx::tensor<Scalar>(dst, new_shape);
  }

  tensor<Scalar> slice(size_t dim, std::size_t begin, std::size_t end) {
    std::vector<std::size_t> v_begin(rank());
    std::vector<std::size_t> v_end(rank());
    assert(dim < rank());
    for (std::size_t d = 0; d < rank(); d++) {
      v_begin[d] = 0;
      v_end[d] = shape()[d] - 1;
    }
    v_begin[dim] = begin;
    v_end[dim] = end;
    return slice(v_begin, v_end);
  }

  // Matrix operations (rank-2 specific)
  tensor<Scalar> dot(const tensor<Scalar> &other) const {

    if (rank() == 2 && other.rank() == 1)
      return mv_product(other);

    if (rank() != 2 || other.rank() != 2) {
      throw std::runtime_error("Dot product requires rank-2 tensors");
    }
    if (shape()[1] != other.shape()[0]) {
      throw std::runtime_error("Invalid matrix dimensions for dot product");
    }

    std::vector<std::size_t> result_shape = {shape()[0], other.shape()[1]};
    tensor<Scalar> result(result_shape);
    pimpl->matrix_dot(other.pimpl.get(), result.pimpl.get());
    return result;
  }

  tensor<Scalar> transpose() const {
    if (rank() != 2) {
      throw std::runtime_error("Transpose requires rank-2 tensors");
    }

    std::vector<std::size_t> result_shape = {shape()[1], shape()[0]};
    tensor<Scalar> result(result_shape);
    pimpl->matrix_transpose(result.pimpl.get());
    return result;
  }

  /// @brief Get a pointer to the raw data of the tensor.
  ///
  /// This method provides direct access to the underlying data storage of the
  /// tensor. It returns a pointer to the first element of the data array.
  ///
  /// @return scalar_type* A pointer to the mutable data of the tensor.
  ///
  /// @note Care should be taken when directly manipulating the raw data to
  /// avoid
  ///       invalidating the tensor's internal state or violating its
  ///       invariants.
  scalar_type *data() { return pimpl->data(); }

  /// @brief Get a const pointer to the raw data of the tensor.
  ///
  /// This method provides read-only access to the underlying data storage of
  /// the tensor. It returns a const pointer to the first element of the data
  /// array.
  ///
  /// @return const scalar_type * A const pointer to the immutable data of the
  /// tensor.
  ///
  /// @note This const version ensures that the tensor's data cannot be modified
  ///       through the returned pointer, preserving const correctness.
  const scalar_type *data() const { return pimpl->data(); }

  void dump() const { pimpl->dump(); }

  /// @brief Dump tensor as bits, where non-zero elements are shown as '1' and
  /// zero-elements are shown as '.'.
  void dump_bits() const { pimpl->dump_bits(); }
};

} // namespace cudaqx
