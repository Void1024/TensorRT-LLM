/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass_extensions/arch/mma.h"

#include "cutlass_extensions/gemm/threadblock/dq_mma_multistage.h"
#include "cutlass_extensions/gemm/warp/default_mma_tensor_op.h"
#include "cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h"
#include "cutlass_extensions/tile_interleaved_layout.h"

#include "cutlass_extensions/gemm/threadblock/default_dq_mma.h"
#include "cutlass_extensions/transform/threadblock/fine_grained_scale_zero_iterator.h"

namespace cutlass
{
namespace gemm
{
namespace threadblock
{

////////////////////////////////////////////////////////////////////////////////

template <typename MmaShape, typename Element, typename Layout, WeightOnlyQuantOp QuantOp, int Alignment,
    typename Enable = void>
struct DefaultScaleIteratorsMultistage;

// Fine grained iterators
template <typename MmaShape, typename Element, typename Layout, WeightOnlyQuantOp QuantOp, int Alignment>
struct DefaultScaleIteratorsMultistage<MmaShape, Element, Layout, QuantOp, Alignment,
    std::enable_if_t<isFinegrained(QuantOp)>>
{
    using IteratorScale
        = cutlass::transform::threadblock::FineGrainedScaleZeroIterator<cutlass::MatrixShape<1, MmaShape::kN>, Element,
            Layout, 0, Alignment>;

    using SmemIteratorScale = IteratorScale;
};

// Per column iterators
template <typename MmaShape, typename Element, typename Layout, WeightOnlyQuantOp QuantOp, int Alignment>
struct DefaultScaleIteratorsMultistage<MmaShape, Element, Layout, QuantOp, Alignment,
    std::enable_if_t<!isFinegrained(QuantOp)>>
{
    // ThreadMap for scale iterator
    static_assert((MmaShape::kN % Alignment) == 0, "");

private:
    using IteratorScaleThreadMap = transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaShape::kN, 1>,
        MmaShape::kN / Alignment, Alignment>;

public:
    // Define iterators over tiles from the scale operand
    using IteratorScale = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaShape::kN>,
        Element, Layout, 0, IteratorScaleThreadMap, Alignment>;

    using SmemIteratorScale = IteratorScale;
};

////////////////////////////////////////////////////////////////////////////////

template <
    /// Type for element A
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Stages in GEMM
    int kStages,
    /// Operator performed by GEMM
    typename Operator_,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear>
struct DqMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementScale, LayoutScale, kAlignmentScale,
    ElementAccumulator, layout::RowMajor, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
    kStages, Operator_, SharedMemoryClear,
    typename platform::enable_if<(
        ArchTag::kMinComputeCapability >= 80 && !layout::IsColumnMajorTileInterleave<LayoutB>::value)>::type>
{

    static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value
            || platform::is_same<ElementA, float_e4m3_t>::value,
        "Element A must be fp16, fp8 or bf16");

    using OperatorInfo = arch::DetagOperator<Operator_>;
    using Operator = typename OperatorInfo::Operator;
    static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
        "Mma multistage must dequantize after ldsm");

    static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
        "Element B must be uint8 or uint4");

    static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
        ? cutlass::arch::CacheOperation::Global
        : cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
        ? cutlass::arch::CacheOperation::Global
        : cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    // Mma core does not depend on stages, so pass in at least 3 here to mma multistage pieces are created
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
        ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator, layout::RowMajor, OperatorClass, std::max(kStages, 3),
        Operator, false, CacheOpA, CacheOpB>;

    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>, ElementA, LayoutA, 1, ThreadMapA,
        AccessTypeA>;

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>, ElementB, LayoutB, 0, ThreadMapB,
        AccessTypeB>;

    using ScaleIterators = DefaultScaleIteratorsMultistage<typename MmaCore::Shape, ElementScale, LayoutScale,
        OperatorInfo::QuantOp, kAlignmentScale>;

    // Define iterators over tiles from the scale operand
    using IteratorScale = typename ScaleIterators::IteratorScale;

    using SmemIteratorScale = typename ScaleIterators::SmemIteratorScale;

    using Converter = FastInterleavedAndBiasedNumericArrayConverter<ElementScale, ElementB,
        MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape, IteratorA,
        typename MmaCore::SmemIteratorA, MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
        MmaCore::kCacheOpB, IteratorScale, SmemIteratorScale, ElementAccumulator, layout::RowMajor,
        typename MmaCore::MmaPolicy, kStages, Converter, OperatorInfo::QuantOp, SharedMemoryClear>;
};
template <
    /// Shape of threadblock-scoped matrix multiply operator
    typename Shape,
    /// Shape of warp-level matrix multiply operator
    typename WarpShape,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape,
    /// Element data type of A operand
    typename ElementA,
    /// Layout of operand A
    typename LayoutA,
    /// Element data type of B operand
    typename ElementB,
    /// Layout of operand B
    typename LayoutB,
    /// Data type of accumulator
    typename ElementC,
    /// Layout of accumulator
    typename LayoutC,
    /// Indicates type of math operator (arch::OpClassSimt or arch::OpClassTensorOp)
    typename OperatorClass,
    /// Number of stages
    int Stages = 2,
    /// Operation performed by MMA
    typename Operator = typename platform::conditional<
        (platform::is_same<OperatorClass,
                           cutlass::arch::OpClassTensorOp>::value) &&
            (platform::is_same<ElementA, int8_t>::value ||
             platform::is_same<ElementA, int4b_t>::value ||
             platform::is_same<ElementA, uint8_t>::value ||
             platform::is_same<ElementA, uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA =
        cutlass::arch::CacheOperation::Global,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB =
        cutlass::arch::CacheOperation::Global,
    /// per-element transformation for elements of A
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// per-element transformation for elements of B
    ComplexTransform TransformB = ComplexTransform::kNone,
    bool IsComplex = false // (is_complex<ElementA>::value || is_complex<ElementB>::value)
>
struct DefaultMmaCoreExt;
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Operation performed by MMA
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCoreExt<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                      Operator_, false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kWarpThreadArrangementContiguousA =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  static int const kWarpThreadArrangementContiguousB =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementB>::value);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kK>;

  // Shared memory layout
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, Shape::kK>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, 128,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, 128,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};
// Specialization to handle column major interleave B
template <
    /// Type for element A
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Stages in GEMM
    int kStages,
    /// Operator performed by GEMM
    typename Operator_,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear>
struct DqMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementScale, LayoutScale, kAlignmentScale,
    ElementAccumulator, layout::RowMajor, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
    kStages, Operator_, SharedMemoryClear,
    typename platform::enable_if<(
        ArchTag::kMinComputeCapability >= 80 && layout::IsColumnMajorTileInterleave<LayoutB>::value)>::type>
{

    static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value
            || platform::is_same<ElementA, float_e4m3_t>::value,
        "Element A must be fp16, fp8 or bf16");

    using OperatorInfo = arch::DetagOperator<Operator_>;
    using Operator = typename OperatorInfo::Operator;
    static_assert(platform::is_same<Operator, arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
        "Mma multistage must dequantize after ldsm");

    static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
        "Element B must be uint8 or uint4");

    static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
        ? cutlass::arch::CacheOperation::Global
        : cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
        ? cutlass::arch::CacheOperation::Global
        : cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    // Mma core does not depend on stages, so pass in at least 3 here to mma multistage pieces are created
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCoreExt<ThreadblockShape, WarpShape, InstructionShape,
        ElementA, LayoutA, ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        std::max(kStages, 3), Operator, false, CacheOpA, CacheOpB>;

    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>, ElementA, LayoutA, 1, ThreadMapA,
        AccessTypeA>;

private:
    static constexpr int ColumnsInterleaved = LayoutB::kColumnsInterleaved;
    static constexpr int RowsPerTile = LayoutB::kRowsPerTile;
    static_assert(!(MmaCore::Shape::kN % ColumnsInterleaved), "");
    static_assert(RowsPerTile == MmaCore::Shape::kK, "");

    using OriginalThreadMap = typename MmaCore::IteratorThreadMapB;
    using OriginalWarpArrangement = typename OriginalThreadMap::Detail::WarpThreadArrangement;
    static_assert(!(OriginalWarpArrangement::kStrided % ColumnsInterleaved), "");

    using GmemIteratorShape
        = MatrixShape<MmaCore::Shape::kK * ColumnsInterleaved, MmaCore::Shape::kN / ColumnsInterleaved>;
    using GmemThreadMapB = transform::PitchLinearWarpRakedThreadMap<
        layout::PitchLinearShape<GmemIteratorShape::kRow, GmemIteratorShape::kColumn>, OriginalThreadMap::kThreads,
        layout::PitchLinearShape<OriginalWarpArrangement::kContiguous * ColumnsInterleaved,
            OriginalWarpArrangement::kStrided / ColumnsInterleaved>,
        MmaCore::kAccessSizeInBits / sizeof_bits<ElementB>::value>;

public:
    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<GmemIteratorShape, ElementB,
        layout::ColumnMajor, 0, GmemThreadMapB, AccessTypeB>;

    using ScaleIterators = DefaultScaleIteratorsMultistage<typename MmaCore::Shape, ElementScale, LayoutScale,
        OperatorInfo::QuantOp, kAlignmentScale>;

    // Define iterators over tiles from the scale operand
    using IteratorScale = typename ScaleIterators::IteratorScale;

    using SmemIteratorScale = typename ScaleIterators::SmemIteratorScale;

    using Converter = FastInterleavedAndBiasedNumericArrayConverter<ElementScale, ElementB,
        MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<typename MmaCore::Shape, IteratorA,
        typename MmaCore::SmemIteratorA, MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
        MmaCore::kCacheOpB, IteratorScale, SmemIteratorScale, ElementAccumulator, layout::RowMajor,
        typename MmaCore::MmaPolicy, kStages, Converter, OperatorInfo::QuantOp, SharedMemoryClear>;
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
