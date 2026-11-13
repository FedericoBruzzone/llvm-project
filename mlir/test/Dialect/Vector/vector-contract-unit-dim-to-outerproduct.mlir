// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// A unit dim (d0) in the middle of the acc (e.g., C viewed as [Mo, m0, No, n0]) is
// cast away, then the contract lowers to a single outer product.

// CHECK-LABEL: func.func @contract_with_unit_dim_to_outerproduct
//  CHECK-SAME:   %[[LHS:.*]]: vector<1x12xf32>, %[[RHS:.*]]: vector<1x1x8xf32>, %[[ACC:.*]]: vector<12x1x8xf32>
//   CHECK-DAG:   %[[ACC_SC:.+]] = vector.shape_cast %[[ACC]] : vector<12x1x8xf32> to vector<12x8xf32>
//   CHECK-DAG:   %[[L:.+]] = vector.extract %[[LHS]][0] : vector<12xf32> from vector<1x12xf32>
//   CHECK-DAG:   %[[R:.+]] = vector.extract %[[RHS]][0, 0] : vector<8xf32> from vector<1x1x8xf32>
//       CHECK:   %[[OP:.+]] = vector.outerproduct %[[L]], %[[R]], %[[ACC_SC]] {kind = #vector.kind<add>} : vector<12xf32>, vector<8xf32>
//       CHECK:   %[[RES:.+]] = vector.shape_cast %[[OP]] : vector<12x8xf32> to vector<12x1x8xf32>
//       CHECK:   return %[[RES]] : vector<12x1x8xf32>
//   CHECK-NOT:   vector.outerproduct
func.func @contract_with_unit_dim_to_outerproduct(%lhs: vector<1x12xf32>, %rhs: vector<1x1x8xf32>, %acc: vector<12x1x8xf32>) -> vector<12x1x8xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d1, d0, d2)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc : vector<1x12xf32>, vector<1x1x8xf32> into vector<12x1x8xf32>
  return %0 : vector<12x1x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.yield
  }
}
