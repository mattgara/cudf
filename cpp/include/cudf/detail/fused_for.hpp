/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/packed_types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cudf::detail {

/**
 * @brief Describes one independently encoded segment of a fused
 * frame-of-reference payload.
 *
 * Byte-aligned segments store `wire_width` bytes per unsigned offset.
 * Bitpacked segments store `wire_width` bits per unsigned offset and set
 * `fused_for_bitpacked`. Each logical value is reconstructed as
 * `reference + offset`, modulo the logical width. Raw segments use equal
 * logical and wire byte widths, a zero reference, and no flags. All offsets
 * are byte offsets.
 */
inline constexpr std::uint16_t fused_for_bitpacked = 1;

struct alignas(8) fused_for_segment {
  std::uint64_t wire_offset;
  std::uint64_t logical_offset;
  std::uint64_t reference;
  std::uint32_t element_count;
  std::uint8_t logical_width;
  std::uint8_t wire_width;
  std::uint16_t flags;

  [[nodiscard]] bool is_bitpacked() const { return (flags & fused_for_bitpacked) != 0; }
  [[nodiscard]] bool is_encoded() const { return is_bitpacked() || wire_width < logical_width; }
};
static_assert(sizeof(fused_for_segment) == 32);

/**
 * @brief A wire payload whose eligible numeric segments were narrowed while
 * the input table was being packed.
 *
 * `metadata` describes the fully reconstructed logical packed table.
 * `wire_data` begins with `segment_count` device-resident segment descriptors,
 * padded to a 16-byte boundary, followed by the encoded bytes. Keeping the
 * descriptors in the transmitted device buffer avoids a host readback per
 * encoded tile. `logical_data_size` is the output allocation size needed by
 * `decode_fused_for`.
 */
struct fused_for_packed_columns {
  std::unique_ptr<std::vector<std::uint8_t>> metadata;
  std::unique_ptr<rmm::device_buffer> wire_data;
  std::size_t segment_count;
  std::size_t logical_data_size;
};

/**
 * @brief Split and pack a table while applying tiled frame-of-reference
 * narrowing to eligible top-level numeric columns.
 *
 * This is intentionally a separate API from `contiguous_split`. The ordinary
 * path remains byte-for-byte unchanged when this function is not called.
 *
 * The transform is fused into the copy that materializes each packed
 * partition. A numeric column is left raw unless its offsets fit in a smaller
 * byte-aligned unsigned storage type.
 *
 * @param input Input table
 * @param splits Row indices at which to split `input`
 * @param stream CUDA stream used for all work
 * @param mr Device memory resource for returned packed allocations
 * @return One compressed wire payload per partition
 */
CUDF_EXPORT std::vector<fused_for_packed_columns> contiguous_split_fused_for(
  table_view const& input,
  std::vector<size_type> const& splits,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Pack a table with the same fused frame-of-reference transform.
 */
CUDF_EXPORT fused_for_packed_columns pack_fused_for(table_view const& input,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

/**
 * @brief Reconstruct ordinary packed columns from a fused
 * frame-of-reference wire payload.
 *
 * The output is materialized exactly once. Raw bytes, null masks, and string
 * offsets are copied into their final packed locations while encoded numeric
 * values are reconstructed directly into their final logical-width locations.
 *
 * @param input Payload produced by `contiguous_split_fused_for`
 * @param stream CUDA stream used for all work
 * @param mr Device memory resource for the reconstructed packed allocation
 */
CUDF_EXPORT packed_columns decode_fused_for(fused_for_packed_columns&& input,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
