#pragma once
namespace cutlass::epilogue::threadblock
{
    
namespace detail {

/// RowArrangement determines how one or more warps cover a region of consecutive rows.
template <
  typename Shape,
  int WarpsRemaining,
  int ElementsPerAccess,
  int ElementSize,
  bool Is2dTile
>
struct RowArrangementExt;
/// RowArrangement in which each warp's access is a 2D tiled arrangement.
template <
  typename Shape,
  int WarpsRemaining,
  int ElementsPerAccess,
  int ElementSize
>
struct RowArrangementExt<Shape, WarpsRemaining, ElementsPerAccess, ElementSize, true> {

  static int const kMemoryAccessSize = 256; // Preferred access size
  static int const kWarpSize = 32;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kElementSize = ElementSize;

  struct Detail {
    static int const kShapeRow = Shape::kRow / WarpsRemaining;
    static_assert(kShapeRow == 2);
    static int const kShapeWidth = Shape::kColumn / kElementsPerAccess;
    static_assert(kShapeWidth == 16);

    static int const kTargetMemoryAccessWidth = 
      kMemoryAccessSize / (kElementsPerAccess * kElementSize / 8);
    static_assert(kTargetMemoryAccessWidth == 16);

    static int const kTargetAccessRows = kWarpSize / kTargetMemoryAccessWidth;
    static_assert(kTargetAccessRows == 2);
  };

  static int const kAccessWidth = 
    (Detail::kTargetAccessRows > Detail::kShapeRow ?
      kWarpSize / Detail::kShapeRow
      : const_min(
          Detail::kShapeWidth,
        const_min(kWarpSize, Detail::kTargetMemoryAccessWidth)
        ));
  static_assert(kAccessWidth == 16);

  static int const kAccessRows =
    (Detail::kTargetAccessRows > Detail::kShapeRow ?
      Detail::kShapeRow
      : const_min(Shape::kRow, kWarpSize / kAccessWidth));
  static_assert(kAccessRows == 2);
  
  static int const kIterationsRow = Detail::kShapeRow / kAccessRows;
  static int const kDeltaRow = kAccessRows;

  static int const kIterationsColumn = Detail::kShapeWidth / kAccessWidth;
  static int const kDeltaColumn = kAccessWidth * kElementsPerAccess;

  static_assert( kAccessWidth * kElementsPerAccess <= Shape::kColumn, "Accessing too many elements per access");
  static_assert( kIterationsColumn > 0, "Iteration Count Column must be > 0" );
  static_assert( kIterationsRow > 0, "Iteration Count Row must be > 0" );

  static int const kWarpPartitionsRow = 1;
  static int const kWarpPartitionsColumn = 1;
};
}
template <
  typename Shape_,
  typename Count_,
  int Threads,
  int ElementsPerAccess,
  int ElementSize
>
struct OutputTileOptimalThreadMapExt {

  using Shape = Shape_;
  using Count = Count_;

  static int const kWarpSize = 32;
  static int const kThreads = Threads;
  static int const kWarpCount = kThreads / kWarpSize;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kElementSize = ElementSize;

  //
  // Metaprogram computation
  //

  struct Detail {

    // Clusters
    static int const kIterationsCluster = 
      ((Shape::kCluster > kWarpCount) ?
        Shape::kCluster / kWarpCount
        : 1);
    static_assert(kIterationsCluster == 1);

    static int const kDeltaCluster =
      ((Shape::kCluster > kWarpCount) ?
        Shape::kRow * Count::kRow * Shape::kGroup * Count::kGroup * Shape::kCluster / kIterationsCluster
        : 1);
    static_assert(kDeltaCluster == 1);

    static int const kCompactedDeltaCluster =
      ((Shape::kCluster > kWarpCount) ?
        Shape::kRow * Shape::kGroup * Shape::kCluster / kIterationsCluster
        : 1);
    static_assert(kCompactedDeltaCluster == 1);

    static int const kWarpPartitionsCluster =
      ((Shape::kCluster > kWarpCount) ?
        kWarpCount
        : kWarpCount / Shape::kCluster);
    static_assert(kWarpPartitionsCluster == kWarpCount);

    static int const kWarpsRemainingForGroups =
      ((Shape::kCluster > kWarpCount) ? 1 : kWarpCount / Shape::kCluster);
    static_assert(kWarpsRemainingForGroups == kWarpCount);

    // Groups
    static int const kIterationsGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        Shape::kGroup / kWarpsRemainingForGroups
        : 1);
    static_assert(kIterationsGroup == 1);

    static int const kDeltaGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        Shape::kRow * Count::kRow * Shape::kGroup / kIterationsGroup
        : 1);
    static_assert(kDeltaGroup == 1);

    static int const kCompactedDeltaGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        Shape::kRow * Shape::kGroup / kIterationsGroup
        : 1);
    static_assert(kCompactedDeltaGroup == 1);

    static int const kWarpPartitionsGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        1
        : kWarpsRemainingForGroups / Shape::kGroup);
    static_assert(kWarpPartitionsGroup == kWarpCount);

    static int const kWarpsRemainingForRows =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        1
        : kWarpsRemainingForGroups / Shape::kGroup);
    static_assert(kWarpsRemainingForRows == kWarpCount);
    
    // Rows
    using RowArrangement = detail::RowArrangementExt<
      Shape,
      kWarpsRemainingForRows,
      kElementsPerAccess,
      kElementSize,
      (Shape::kRow > kWarpsRemainingForRows)
    >;

    // Warp partitions
    using WarpPartitions = OutputTileShape<
      RowArrangement::kWarpPartitionsColumn,
      RowArrangement::kWarpPartitionsRow,
      kWarpPartitionsGroup,
      kWarpPartitionsCluster,
      1>;

    static int const kAccessWidth = RowArrangement::kAccessWidth;
    static int const kAccessRows = RowArrangement::kAccessRows;
  };

  //
  // Output
  //

  using Iterations = OutputTileShape<
    Detail::RowArrangement::kIterationsColumn, 
    Detail::RowArrangement::kIterationsRow, 
    Detail::kIterationsGroup, 
    Detail::kIterationsCluster, 
    1>;

  using Delta = OutputTileShape<
    Detail::RowArrangement::kDeltaColumn,
    Detail::RowArrangement::kDeltaRow,
    Detail::kDeltaGroup,
    Detail::kDeltaCluster,
    1>;

  /// Initial offset function
  CUTLASS_DEVICE
  static MatrixCoord initial_offset(int thread_idx) {
    int warp_idx = thread_idx / kWarpSize;
    int lane_idx = thread_idx % kWarpSize;

    // Compute warp location
    int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
    int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

    int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
    int residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

    int row_idx = residual_group / Detail::WarpPartitions::kRow;
    int col_idx = residual_group % Detail::WarpPartitions::kRow;

    // Compute per-lane offset
    int lane_row_offset = lane_idx / Detail::kAccessWidth;
    int lane_col_offset = lane_idx % Detail::kAccessWidth;

    // Compute coordinate in output space
    int cluster_offset = cluster_idx * Shape::kRow * Count::kRow * Shape::kGroup * Count::kGroup;
    int group_offset = group_idx * Shape::kRow * Count::kRow;
    int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
    int column_offset = col_idx * Iterations::kColumn * Detail::kAccessWidth * kElementsPerAccess;

    return MatrixCoord(
      cluster_offset + group_offset + row_offset + lane_row_offset,
      column_offset + lane_col_offset * kElementsPerAccess
    );
  }

  /// Computes the offset of a given vector access
  CUTLASS_HOST_DEVICE
  static MatrixCoord iteration_offset(int iter_idx) {
    return OutputTileThreadMapHelpers<Iterations, Delta>::iteration_offset(iter_idx);
  }

  /// Compacted thread map in which the 4D region is contiguous
  struct CompactedThreadMap {


    using Shape = Shape_;

    using TileShape = MatrixShape<
      Shape::kTile * Shape::kCluster * Shape::kGroup * Shape::kRow,
      Shape::kColumn
    >;

    using Iterations = OutputTileShape<
      Detail::RowArrangement::kIterationsColumn,
      Detail::RowArrangement::kIterationsRow,
      Detail::kIterationsGroup,
      Detail::kIterationsCluster,
      1>;

    using Delta = OutputTileShape<
      Detail::RowArrangement::kDeltaColumn,
      Detail::RowArrangement::kDeltaRow,
      Detail::kCompactedDeltaGroup,
      Detail::kCompactedDeltaCluster,
      1>;

    /// Number of elements within each vector access
    static int const kElementsPerAccess = ElementsPerAccess;

    /// Number  of threads
    static int const kThreads = Threads;

    /// Function to compute each thread's initial offset
    CUTLASS_DEVICE
    static MatrixCoord initial_offset(int thread_idx) {
      int warp_idx = thread_idx / kWarpSize;
      int lane_idx = thread_idx % kWarpSize;

      // Compute warp location
      int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
      int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

      int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
      int residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

      int row_idx = residual_group / Detail::WarpPartitions::kRow;
      int col_idx = residual_group % Detail::WarpPartitions::kRow;

      // Compute per-lane offset
      int lane_row_offset = lane_idx / Detail::kAccessWidth;
      int lane_col_offset = lane_idx % Detail::kAccessWidth;

      // Compute coordinate in output space
      int cluster_offset = cluster_idx * Shape::kRow * Shape::kGroup;
      int group_offset = group_idx * Shape::kRow;
      int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
      int column_offset = col_idx * Iterations::kColumn * Detail::kAccessWidth * kElementsPerAccess;

      MatrixCoord coord(
        cluster_offset + group_offset + row_offset + lane_row_offset,
        column_offset + lane_col_offset * kElementsPerAccess
      );

      return coord;
    }
  };
};
template <int CtaShapeM>
struct DefaultThreadMapTensorOp<cutlass::gemm::GemmShape<CtaShapeM, 128, 64>, cutlass::gemm::GemmShape<CtaShapeM, 32, 32>, 2, cutlass::half_t, 8>
{
    using ThreadblockShape = cutlass::gemm::GemmShape<CtaShapeM, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<CtaShapeM, 32, 32>;
    static int const kPartitionsK = 2;
    using Element = cutlass::half_t;
    static int const kElementsPerAccess = 8;

    struct Detail
    {
        static int const kTensorOpRows = 8;
        static int const kWarpSize = 32;
        static_assert(
            !(ThreadblockShape::kM % WarpShape::kM) && !(ThreadblockShape::kN % WarpShape::kN), "Divisibility");
        using WarpCount
            = gemm::GemmShape<ThreadblockShape::kM / WarpShape::kM, ThreadblockShape::kN / WarpShape::kN, kPartitionsK>;
        static int const kThreads = WarpCount::kCount * kWarpSize;
    };

    using Type = OutputTileOptimalThreadMapExt<
        OutputTileShape<ThreadblockShape::kN, Detail::kTensorOpRows, Detail::WarpCount::kM, 1, 1>,
        OutputTileShape<1, WarpShape::kM / Detail::kTensorOpRows, 1, 1, WarpShape::kM / Detail::kTensorOpRows>,
        128, kElementsPerAccess, sizeof_bits<Element>::value>;
};
}; // namespace cutlass::epilogue::threadblock