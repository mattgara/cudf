/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/fused_for.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

struct FusedForSplitTest : public cudf::test::BaseFixture {};

std::vector<std::uint8_t> copy_to_host(rmm::device_buffer const& input, cuda::stream_ref stream)
{
  std::vector<std::uint8_t> result(input.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    result.data(), input.data(), input.size(), cudaMemcpyDeviceToHost, stream.get()));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return result;
}

std::vector<cudf::detail::fused_for_segment> copy_descriptors_to_host(
  cudf::detail::fused_for_packed_columns const& input, cuda::stream_ref stream)
{
  std::vector<cudf::detail::fused_for_segment> result(input.segment_count);
  if (result.empty()) { return result; }

  CUDF_CUDA_TRY(cudaMemcpyAsync(result.data(),
                                input.wire_data->data(),
                                result.size() * sizeof(result.front()),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return result;
}

void expect_packed_equal(cudf::packed_columns const& expected,
                         cudf::packed_columns const& actual,
                         cuda::stream_ref stream)
{
  ASSERT_NE(expected.metadata, nullptr);
  ASSERT_NE(actual.metadata, nullptr);
  ASSERT_NE(expected.gpu_data, nullptr);
  ASSERT_NE(actual.gpu_data, nullptr);
  EXPECT_EQ(*expected.metadata, *actual.metadata);
  EXPECT_EQ(expected.gpu_data->size(), actual.gpu_data->size());
  EXPECT_EQ(copy_to_host(*expected.gpu_data, stream), copy_to_host(*actual.gpu_data, stream));
}

void expect_fused_for_round_trip(cudf::table_view const& input,
                                 std::vector<cudf::size_type> const& splits)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto expected     = cudf::contiguous_split(input, splits, stream, mr);
  auto packed       = cudf::detail::contiguous_split_fused_for(input, splits, stream, mr);

  ASSERT_EQ(expected.size(), packed.size());
  for (std::size_t i = 0; i < packed.size(); ++i) {
    auto decoded = cudf::detail::decode_fused_for(std::move(packed[i]), stream, mr);
    expect_packed_equal(expected[i].data, decoded, stream);
  }
}

TEST_F(FusedForSplitTest, UsesAllByteAlignedWidthsAndPreservesRawColumns)
{
  constexpr cudf::size_type rows = 10'000;
  std::vector<std::int16_t> one_byte(rows);
  std::vector<std::int32_t> two_bytes(rows);
  std::vector<std::int64_t> four_bytes(rows);
  std::vector<std::int64_t> eight_bytes(rows);
  std::vector<float> floating(rows);
  std::vector<std::string> strings(rows);
  std::vector<bool> valid(rows);

  for (cudf::size_type i = 0; i < rows; ++i) {
    one_byte[i]    = i % 2 == 0 ? -100 : 100;
    two_bytes[i]   = i % 2 == 0 ? -30'000 : 30'000;
    four_bytes[i]  = i % 2 == 0 ? -1'000'000'000LL : 1'000'000'000LL;
    eight_bytes[i] = i % 2 == 0 ? std::numeric_limits<std::int64_t>::lowest()
                                : std::numeric_limits<std::int64_t>::max();
    floating[i]    = static_cast<float>(i) * 0.25F;
    strings[i]     = i % 2 == 0 ? "alpha" : "beta-value";
    valid[i]       = i % 7 != 0;
  }

  cudf::test::fixed_width_column_wrapper<std::int16_t> byte_col(one_byte.begin(), one_byte.end());
  cudf::test::fixed_width_column_wrapper<std::int32_t> short_col(two_bytes.begin(),
                                                                 two_bytes.end());
  cudf::test::fixed_width_column_wrapper<std::int64_t> word_col(four_bytes.begin(),
                                                                four_bytes.end());
  cudf::test::fixed_width_column_wrapper<std::int64_t> raw_col(
    eight_bytes.begin(), eight_bytes.end(), valid.begin());
  cudf::test::fixed_width_column_wrapper<float> float_col(floating.begin(), floating.end());
  cudf::test::strings_column_wrapper string_col(strings.begin(), strings.end());
  cudf::table_view input{{byte_col, short_col, word_col, raw_col, float_col, string_col}};

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto packed       = cudf::detail::pack_fused_for(input, stream, mr);
  auto descriptors  = copy_descriptors_to_host(packed, stream);

  auto has_width = [&descriptors](std::uint8_t logical, std::uint8_t wire) {
    return std::any_of(descriptors.begin(), descriptors.end(), [=](auto const& segment) {
      return segment.logical_width == logical && segment.wire_width == wire;
    });
  };
  EXPECT_TRUE(has_width(2, 1));
  EXPECT_TRUE(has_width(4, 2));
  EXPECT_TRUE(has_width(8, 4));
  EXPECT_TRUE(has_width(8, 8));
  EXPECT_LT(packed.wire_data->size(), packed.logical_data_size);

  auto expected = cudf::pack(input, stream, mr);
  auto decoded  = cudf::detail::decode_fused_for(std::move(packed), stream, mr);
  expect_packed_equal(expected, decoded, stream);
}

TEST_F(FusedForSplitTest, RawSegmentsReserveWireSpaceOncePerTile)
{
  constexpr cudf::size_type rows = 65'536;
  std::vector<float> values(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    values[i] = static_cast<float>(i) * 0.25F;
  }

  cudf::test::fixed_width_column_wrapper<float> column(values.begin(), values.end());
  cudf::table_view input{{column}};
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto packed       = cudf::detail::pack_fused_for(input, stream, mr);
  auto descriptors  = copy_descriptors_to_host(packed, stream);

  ASSERT_GT(descriptors.size(), 1);
  auto round_up_16 = [](std::size_t bytes) { return (bytes + 15) & ~std::size_t{15}; };
  auto expected_wire_size =
    round_up_16(descriptors.size() * sizeof(cudf::detail::fused_for_segment));
  for (auto const& segment : descriptors) {
    EXPECT_EQ(segment.logical_width, 1);
    EXPECT_EQ(segment.wire_width, 1);
    auto const segment_bytes = static_cast<std::size_t>(segment.element_count);
    ASSERT_LE(segment.wire_offset, packed.wire_data->size());
    EXPECT_LE(segment_bytes, packed.wire_data->size() - segment.wire_offset);
    expected_wire_size += round_up_16(segment_bytes);
  }
  EXPECT_EQ(packed.wire_data->size(), expected_wire_size);

  auto expected = cudf::pack(input, stream, mr);
  auto decoded  = cudf::detail::decode_fused_for(std::move(packed), stream, mr);
  expect_packed_equal(expected, decoded, stream);
}

TEST_F(FusedForSplitTest, NonDefaultStreamPreservesAsyncDecodeLifetime)
{
  rmm::cuda_stream stream;
  auto const stream_view = stream.view();
  auto const mr          = cudf::get_current_device_resource_ref();

  constexpr cudf::size_type rows = 16'384;
  std::vector<std::int64_t> values(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    values[i] = -500'000 + (i % 2'048);
  }

  cudf::test::fixed_width_column_wrapper<std::int64_t> column(values.begin(), values.end());
  cudf::table_view input{{column}};
  auto expected = cudf::pack(input, stream_view, mr);
  auto packed   = cudf::detail::pack_fused_for(input, stream_view, mr);

  // decode_fused_for launches asynchronously and destroys the moved wire
  // buffer on return. Its deallocation must remain ordered after the decode
  // on this non-default stream.
  auto decoded = cudf::detail::decode_fused_for(std::move(packed), stream_view, mr);
  expect_packed_equal(expected, decoded, stream_view);
}

TEST_F(FusedForSplitTest, RoundTripsNullableSliceAcrossPartitions)
{
  constexpr cudf::size_type rows = 25'000;
  std::vector<std::int64_t> values(rows);
  std::vector<bool> valid(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    values[i] = -4'000'000 + (i % 4'000);
    valid[i]  = i % 11 != 0;
  }

  cudf::test::fixed_width_column_wrapper<std::int64_t> column(
    values.begin(), values.end(), valid.begin());
  cudf::table_view original{{column}};
  auto sliced = cudf::slice(original, {17, rows - 19});
  ASSERT_EQ(sliced.size(), 1);

  expect_fused_for_round_trip(sliced.front(), {4'000, 12'000, 20'000});
}

TEST_F(FusedForSplitTest, ConstantsAndSignedExtremaRoundTrip)
{
  constexpr cudf::size_type rows = 4'096;
  std::vector<std::int64_t> constants(rows, -9'223'372'036'854'000'000LL);
  std::vector<std::int64_t> extrema(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    extrema[i] = i % 2 == 0 ? std::numeric_limits<std::int64_t>::lowest()
                            : std::numeric_limits<std::int64_t>::max();
  }

  cudf::test::fixed_width_column_wrapper<std::int64_t> constant_col(constants.begin(),
                                                                    constants.end());
  cudf::test::fixed_width_column_wrapper<std::int64_t> extrema_col(extrema.begin(), extrema.end());
  cudf::table_view input{{constant_col, extrema_col}};
  expect_fused_for_round_trip(input, {});
}

TEST_F(FusedForSplitTest, PreservesMetadataOnlyZeroColumnTable)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 7};
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto expected     = cudf::contiguous_split(input, {3}, stream, mr);
  auto packed       = cudf::detail::contiguous_split_fused_for(input, {3}, stream, mr);

  ASSERT_EQ(packed.size(), 2);
  for (std::size_t i = 0; i < packed.size(); ++i) {
    EXPECT_EQ(packed[i].segment_count, 0);
    EXPECT_EQ(packed[i].logical_data_size, 0);
    auto decoded = cudf::detail::decode_fused_for(std::move(packed[i]), stream, mr);
    expect_packed_equal(expected[i].data, decoded, stream);
  }
}

TEST_F(FusedForSplitTest, RejectsMalformedEnvelope)
{
  cudf::test::fixed_width_column_wrapper<std::int64_t> column({1, 2, 3, 4, 5});
  cudf::table_view input{{column}};
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  {
    auto packed          = cudf::detail::pack_fused_for(input, stream, mr);
    packed.segment_count = packed.wire_data->size() / sizeof(cudf::detail::fused_for_segment) + 1;
    EXPECT_THROW(cudf::detail::decode_fused_for(std::move(packed), stream, mr), cudf::logic_error);
  }
  {
    auto packed = cudf::detail::pack_fused_for(input, stream, mr);
    ASSERT_NE(packed.logical_data_size, 0);
    packed.segment_count = 0;
    EXPECT_THROW(cudf::detail::decode_fused_for(std::move(packed), stream, mr), cudf::logic_error);
  }
}
