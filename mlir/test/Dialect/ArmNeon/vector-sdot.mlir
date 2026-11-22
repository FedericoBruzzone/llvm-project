// RUN:  mlir-opt %s --transform-interpreter | FileCheck %s

#mmt = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#vecmat = [
  affine_map<(n, k) -> (k)>,
  affine_map<(n, k) -> (n, k)>,
  affine_map<(n, k) -> (n)>
]

//===----------------------------------------------------------------------===//
// Matrix-matrix: LHS is (m, k), RHS is (n, k), ACC is (m, n).
//
// For each iteration [I, J, K], with a [1, 4, 4] sub-tile, the operand
// sub-tiles are extracted at:
//   LHS: [I, 4*K], viewed as one i32 and broadcast to the four 32-bit lanes
//   RHS: [4*J, 4*K]
//   ACC: [I, 4*J], read only at K = 0, then the SDOT result is chained along K
// The result sub-tile is inserted at the ACC offsets.
//===----------------------------------------------------------------------===//

// A single ACC sub-tile and two K steps: the second SDOT accumulates into the
// first one.

// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:    %[[LHS:.+]]: vector<1x8xi8>, %[[RHS:.+]]: vector<4x8xi8>, %[[ACC:.+]]: vector<1x4xi32>

// Iteration [0, 0, 0]
// CHECK: %[[L0:.+]] = vector.extract_strided_slice %[[LHS]] offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : vector<1x8xi8> to vector<1x4xi8>
// CHECK: %[[R0:.+]] = vector.extract_strided_slice %[[RHS]] offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : vector<4x8xi8> to vector<4x4xi8>
// CHECK: %[[G0:.+]] = vector.shape_cast %[[L0]] : vector<1x4xi8> to vector<4xi8>
// CHECK: %[[I0:.+]] = vector.bitcast %[[G0]] : vector<4xi8> to vector<1xi32>
// CHECK: %[[B0:.+]] = vector.broadcast %[[I0]] : vector<1xi32> to vector<4xi32>
// CHECK: %[[FL0:.+]] = vector.bitcast %[[B0]] : vector<4xi32> to vector<16xi8>
// CHECK: %[[FR0:.+]] = vector.shape_cast %[[R0]] : vector<4x4xi8> to vector<16xi8>
// CHECK: %[[FA:.+]] = vector.shape_cast %[[ACC]] : vector<1x4xi32> to vector<4xi32>
// CHECK: %[[K_ACC_0:.+]] = arm_neon.intr.sdot %[[FA]], %[[FR0]], %[[FL0]] : vector<16xi8>, vector<16xi8> to vector<4xi32>

// Iteration [0, 0, 1]: accumulates into the result of iteration [0, 0, 0]
// CHECK: %[[L1:.+]] = vector.extract_strided_slice %[[LHS]] offsets = [0, 4], sizes = [1, 4], strides = [1, 1] : vector<1x8xi8> to vector<1x4xi8>
// CHECK: %[[R1:.+]] = vector.extract_strided_slice %[[RHS]] offsets = [0, 4], sizes = [4, 4], strides = [1, 1] : vector<4x8xi8> to vector<4x4xi8>
// CHECK: %[[G1:.+]] = vector.shape_cast %[[L1]] : vector<1x4xi8> to vector<4xi8>
// CHECK: %[[I1:.+]] = vector.bitcast %[[G1]] : vector<4xi8> to vector<1xi32>
// CHECK: %[[B1:.+]] = vector.broadcast %[[I1]] : vector<1xi32> to vector<4xi32>
// CHECK: %[[FL1:.+]] = vector.bitcast %[[B1]] : vector<4xi32> to vector<16xi8>
// CHECK: %[[FR1:.+]] = vector.shape_cast %[[R1]] : vector<4x4xi8> to vector<16xi8>
// CHECK: %[[K_ACC_1:.+]] = arm_neon.intr.sdot %[[K_ACC_0]], %[[FR1]], %[[FL1]] : vector<16xi8>, vector<16xi8> to vector<4xi32>
// CHECK: %[[RES:.+]] = vector.shape_cast %[[K_ACC_1]] : vector<4xi32> to vector<1x4xi32>
// CHECK: return %[[RES]] : vector<1x4xi32>
func.func @matmul(%lhs: vector<1x8xi8>,
                  %rhs: vector<4x8xi8>,
                  %acc: vector<1x4xi32>) -> vector<1x4xi32> {
  %0 = vector.contract {indexing_maps = #mmt,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<1x8xi8>, vector<4x8xi8> into vector<1x4xi32>
  return %0 : vector<1x4xi32>
}

// Unrolling over M, N and K: M = 2, N = 2 * 4, K = 2 * 4 gives 2 * 2 * 2 = 8
// SDOTs. Each ACC sub-tile is read once, then two SDOTs are chained along K.

// CHECK-LABEL: func.func @matmul_unroll
// CHECK-SAME:    %[[LHS:.+]]: vector<2x8xi8>, %[[RHS:.+]]: vector<8x8xi8>, %[[ACC:.+]]: vector<2x8xi32>
// CHECK:         vector.extract_strided_slice %[[ACC]] offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : vector<2x8xi32> to vector<1x4xi32>
// CHECK-COUNT-2: arm_neon.intr.sdot
// CHECK:         vector.extract_strided_slice %[[ACC]] offsets = [0, 4], sizes = [1, 4], strides = [1, 1] : vector<2x8xi32> to vector<1x4xi32>
// CHECK-COUNT-2: arm_neon.intr.sdot
// CHECK:         vector.extract_strided_slice %[[ACC]] offsets = [1, 0], sizes = [1, 4], strides = [1, 1] : vector<2x8xi32> to vector<1x4xi32>
// CHECK-COUNT-2: arm_neon.intr.sdot
// CHECK:         vector.extract_strided_slice %[[ACC]] offsets = [1, 4], sizes = [1, 4], strides = [1, 1] : vector<2x8xi32> to vector<1x4xi32>
// CHECK-COUNT-2: arm_neon.intr.sdot
// CHECK-NOT:     arm_neon.intr.sdot
// CHECK:         return
func.func @matmul_unroll(%lhs: vector<2x8xi8>,
                         %rhs: vector<8x8xi8>,
                         %acc: vector<2x8xi32>) -> vector<2x8xi32> {
  %0 = vector.contract {indexing_maps = #mmt,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<2x8xi8>, vector<8x8xi8> into vector<2x8xi32>
  return %0 : vector<2x8xi32>
}

//===----------------------------------------------------------------------===//
// Vector-matrix: LHS is (k), RHS is (n, k), ACC is (n).
//
// Same as the matrix-matrix case without M. For each iteration [J, K], with a
// [4, 4] sub-tile, the operand sub-tiles are extracted at:
//   LHS: [4*K], viewed as one i32 and broadcast to the four 32-bit lanes
//   RHS: [4*J, 4*K]
//   ACC: [4*J]
//===----------------------------------------------------------------------===//

// A single ACC sub-tile and two K steps.

// CHECK-LABEL: func.func @vecmat
// CHECK-SAME:    %[[LHS:.+]]: vector<8xi8>, %[[RHS:.+]]: vector<4x8xi8>, %[[ACC:.+]]: vector<4xi32>
// CHECK: %[[L0:.+]] = vector.extract_strided_slice %[[LHS]] offsets = [0], sizes = [4], strides = [1] : vector<8xi8> to vector<4xi8>
// CHECK: %[[R0:.+]] = vector.extract_strided_slice %[[RHS]] offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : vector<4x8xi8> to vector<4x4xi8>
// CHECK: %[[I0:.+]] = vector.bitcast %[[L0]] : vector<4xi8> to vector<1xi32>
// CHECK: %[[B0:.+]] = vector.broadcast %[[I0]] : vector<1xi32> to vector<4xi32>
// CHECK: %[[FL0:.+]] = vector.bitcast %[[B0]] : vector<4xi32> to vector<16xi8>
// CHECK: %[[FR0:.+]] = vector.shape_cast %[[R0]] : vector<4x4xi8> to vector<16xi8>
// CHECK: %[[K_ACC_0:.+]] = arm_neon.intr.sdot %[[ACC]], %[[FR0]], %[[FL0]] : vector<16xi8>, vector<16xi8> to vector<4xi32>
// CHECK: %[[L1:.+]] = vector.extract_strided_slice %[[LHS]] offsets = [4], sizes = [4], strides = [1] : vector<8xi8> to vector<4xi8>
// CHECK: %[[R1:.+]] = vector.extract_strided_slice %[[RHS]] offsets = [0, 4], sizes = [4, 4], strides = [1, 1] : vector<4x8xi8> to vector<4x4xi8>
// CHECK: %[[I1:.+]] = vector.bitcast %[[L1]] : vector<4xi8> to vector<1xi32>
// CHECK: %[[B1:.+]] = vector.broadcast %[[I1]] : vector<1xi32> to vector<4xi32>
// CHECK: %[[FL1:.+]] = vector.bitcast %[[B1]] : vector<4xi32> to vector<16xi8>
// CHECK: %[[FR1:.+]] = vector.shape_cast %[[R1]] : vector<4x4xi8> to vector<16xi8>
// CHECK: %[[K_ACC_1:.+]] = arm_neon.intr.sdot %[[K_ACC_0]], %[[FR1]], %[[FL1]] : vector<16xi8>, vector<16xi8> to vector<4xi32>
// CHECK: return %[[K_ACC_1]] : vector<4xi32>
func.func @vecmat(%lhs: vector<8xi8>,
                  %rhs: vector<4x8xi8>,
                  %acc: vector<4xi32>) -> vector<4xi32> {
  %0 = vector.contract {indexing_maps = #vecmat,
                        iterator_types = ["parallel", "reduction"],
                        kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<8xi8>, vector<4x8xi8> into vector<4xi32>
  return %0 : vector<4xi32>
}

//===----------------------------------------------------------------------===//
// Sign extensions: SDOT takes i8 operands, so an operand narrower than i8 is
// sign-extended to i8 first.
//===----------------------------------------------------------------------===//

// LHS is i8 and RHS is i4, both explicitly sign-extended to i32.

// CHECK-LABEL: func.func @matmul_rhs_i4
// CHECK-SAME:    %[[LHS:.+]]: vector<1x4xi8>, %[[RHS:.+]]: vector<4x4xi4>, %[[ACC:.+]]: vector<1x4xi32>
// CHECK: %[[R:.+]] = arith.extsi %[[RHS]] : vector<4x4xi4> to vector<4x4xi8>
// CHECK: %[[G:.+]] = vector.shape_cast %[[LHS]] : vector<1x4xi8> to vector<4xi8>
// CHECK: %[[I:.+]] = vector.bitcast %[[G]] : vector<4xi8> to vector<1xi32>
// CHECK: %[[B:.+]] = vector.broadcast %[[I]] : vector<1xi32> to vector<4xi32>
// CHECK: %[[FL:.+]] = vector.bitcast %[[B]] : vector<4xi32> to vector<16xi8>
// CHECK: %[[FR:.+]] = vector.shape_cast %[[R]] : vector<4x4xi8> to vector<16xi8>
// CHECK: %[[FA:.+]] = vector.shape_cast %[[ACC]] : vector<1x4xi32> to vector<4xi32>
// CHECK: %[[S:.+]] = arm_neon.intr.sdot %[[FA]], %[[FR]], %[[FL]] : vector<16xi8>, vector<16xi8> to vector<4xi32>
// CHECK: %[[RES:.+]] = vector.shape_cast %[[S]] : vector<4xi32> to vector<1x4xi32>
// CHECK: return %[[RES]] : vector<1x4xi32>
func.func @matmul_rhs_i4(%lhs: vector<1x4xi8>,
                         %rhs: vector<4x4xi4>,
                         %acc: vector<1x4xi32>) -> vector<1x4xi32> {
  %lhs_ext = arith.extsi %lhs : vector<1x4xi8> to vector<1x4xi32>
  %rhs_ext = arith.extsi %rhs : vector<4x4xi4> to vector<4x4xi32>
  %0 = vector.contract {indexing_maps = #mmt,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
    %lhs_ext, %rhs_ext, %acc : vector<1x4xi32>, vector<4x4xi32> into vector<1x4xi32>
  return %0 : vector<1x4xi32>
}

//===----------------------------------------------------------------------===//
// Negative tests: the contraction is left as it is.
//===----------------------------------------------------------------------===//

// Zero-extended LHS: SDOT is signed only.

// CHECK-LABEL: func.func @negative_lhs_extui
// CHECK:       vector.contract
// CHECK-NOT:   arm_neon.intr.sdot
func.func @negative_lhs_extui(%lhs: vector<1x4xi8>, %rhs: vector<4x4xi8>,
                              %acc: vector<1x4xi32>) -> vector<1x4xi32> {
  %lhs_ext = arith.extui %lhs : vector<1x4xi8> to vector<1x4xi32>
  %rhs_ext = arith.extsi %rhs : vector<4x4xi8> to vector<4x4xi32>
  %0 = vector.contract {indexing_maps = #mmt,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
    %lhs_ext, %rhs_ext, %acc : vector<1x4xi32>, vector<4x4xi32> into vector<1x4xi32>
  return %0 : vector<1x4xi32>
}

// RHS is (k, n), not transposed.

// CHECK-LABEL: func.func @negative_rhs_not_transposed
// CHECK:       vector.contract
// CHECK-NOT:   arm_neon.intr.sdot
func.func @negative_rhs_not_transposed(%lhs: vector<1x4xi8>, %rhs: vector<4x4xi8>,
                                       %acc: vector<1x4xi32>) -> vector<1x4xi32> {
  %0 = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>,
                          affine_map<(m, n, k) -> (k, n)>,
                          affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<1x4xi8>, vector<4x4xi8> into vector<1x4xi32>
  return %0 : vector<1x4xi32>
}

// K is not a multiple of 4.

// CHECK-LABEL: func.func @negative_k_not_multiple_of_4
// CHECK:       vector.contract
// CHECK-NOT:   arm_neon.intr.sdot
func.func @negative_k_not_multiple_of_4(%lhs: vector<1x6xi8>, %rhs: vector<4x6xi8>,
                                        %acc: vector<1x4xi32>) -> vector<1x4xi32> {
  %0 = vector.contract {indexing_maps = #mmt,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<1x6xi8>, vector<4x6xi8> into vector<1x4xi32>
  return %0 : vector<1x4xi32>
}

// N is not a multiple of 4.

// CHECK-LABEL: func.func @negative_n_not_multiple_of_4
// CHECK:       vector.contract
// CHECK-NOT:   arm_neon.intr.sdot
func.func @negative_n_not_multiple_of_4(%lhs: vector<1x4xi8>, %rhs: vector<2x4xi8>,
                                        %acc: vector<1x2xi32>) -> vector<1x2xi32> {
  %0 = vector.contract {indexing_maps = #mmt,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<1x4xi8>, vector<2x4xi8> into vector<1x2xi32>
  return %0 : vector<1x2xi32>
}

// The accumulator is not i32.

// CHECK-LABEL: func.func @negative_acc_not_i32
// CHECK:       vector.contract
// CHECK-NOT:   arm_neon.intr.sdot
func.func @negative_acc_not_i32(%lhs: vector<1x4xi8>, %rhs: vector<4x4xi8>,
                                %acc: vector<1x4xi16>) -> vector<1x4xi16> {
  %0 = vector.contract {indexing_maps = #mmt,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<1x4xi8>, vector<4x4xi8> into vector<1x4xi16>
  return %0 : vector<1x4xi16>
}

// The combining kind is not add.

// CHECK-LABEL: func.func @negative_kind_not_add
// CHECK:       vector.contract
// CHECK-NOT:   arm_neon.intr.sdot
func.func @negative_kind_not_add(%lhs: vector<1x4xi32>, %rhs: vector<4x4xi32>,
                                 %acc: vector<1x4xi32>) -> vector<1x4xi32> {
  %0 = vector.contract {indexing_maps = #mmt,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<maxsi>}
    %lhs, %rhs, %acc : vector<1x4xi32>, vector<4x4xi32> into vector<1x4xi32>
  return %0 : vector<1x4xi32>
}

// Masked contractions must be left alone: rewriting one in place would leave
// more than one op inside the `vector.mask` region, which is invalid.

// CHECK-LABEL: func.func @negative_masked
// CHECK:       vector.mask %{{.*}} { vector.contract
// CHECK-NOT:   arm_neon.intr.sdot
func.func @negative_masked(%lhs: vector<1x4xi8>, %rhs: vector<4x4xi8>,
                           %acc: vector<1x4xi32>, %m: index, %n: index,
                           %k: index) -> vector<1x4xi32> {
  %mask = vector.create_mask %m, %n, %k : vector<1x4x4xi1>
  %0 = vector.mask %mask {
    vector.contract {indexing_maps = #mmt,
                     iterator_types = ["parallel", "parallel", "reduction"],
                     kind = #vector.kind<add>}
      %lhs, %rhs, %acc : vector<1x4xi8>, vector<4x4xi8> into vector<1x4xi32>
  } : vector<1x4x4xi1> -> vector<1x4xi32>
  return %0 : vector<1x4xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func {
      transform.apply_patterns.arm_neon.vector_contract_to_sdot
    } : !transform.op<"func.func">

    transform.yield
  }
}
