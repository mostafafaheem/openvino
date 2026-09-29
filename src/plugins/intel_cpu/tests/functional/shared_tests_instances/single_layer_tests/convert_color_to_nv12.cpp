// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/convert_color_to_nv12.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ConvertColorToNV12LayerTest;

const std::vector<std::vector<ov::Shape>> in_shapes = {{{1, 2, 2, 3}},
                                                       {{1, 4, 4, 3}},
                                                       {{1, 6, 6, 3}},
                                                       {{1, 8, 8, 3}},
                                                       {{1, 10, 10, 3}},
                                                       {{1, 16, 16, 3}},
                                                       {{1, 18, 18, 3}},
                                                       {{1, 20, 22, 3}},
                                                       {{1, 32, 34, 3}},
                                                       {{1, 8, 32, 3}},
                                                       {{1, 32, 8, 3}},
                                                       {{1, 16, 64, 3}},
                                                       {{1, 64, 16, 3}},
                                                       {{2, 8, 8, 3}},
                                                       {{3, 10, 10, 3}},
                                                       {{4, 16, 16, 3}}};

const std::vector<ov::element::Type> inTypes = {ov::element::u8,
                                                ov::element::bf16,
                                                ov::element::f16,
                                                ov::element::f32,
                                                ov::element::f64};
const std::vector<std::vector<ov::Shape>> in_shapes_acc = {{{1, 64, 96, 3}}};

const std::vector<std::vector<ov::Shape>> in_shapes_nightly = {{{1, 256, 256, 3}}, {{1, 720, 1280, 3}}};

const auto test_case_values =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes)),
                       ::testing::ValuesIn(inTypes),
                       ::testing::Bool(),
                       ::testing::Bool(),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorToNV12_Single_Plane,
                         ConvertColorToNV12LayerTest,
                         test_case_values,
                         ConvertColorToNV12LayerTest::getTestCaseName);

const auto testCase_accuracy_values =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_acc)),
                       ::testing::Values(ov::element::u8),
                       ::testing::Bool(),
                       ::testing::Values(true),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorToNV12_acc,
                         ConvertColorToNV12LayerTest,
                         testCase_accuracy_values,
                         ConvertColorToNV12LayerTest::getTestCaseName);

const auto testCase_accuracy_values_nightly =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_nightly)),
                       ::testing::Values(ov::element::u8),
                       ::testing::Bool(),
                       ::testing::Values(true),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(nightly_TestsConvertColorToNV12_acc,
                         ConvertColorToNV12LayerTest,
                         testCase_accuracy_values_nightly,
                         ConvertColorToNV12LayerTest::getTestCaseName);

}  // namespace
