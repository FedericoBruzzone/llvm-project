// REQUIRES: mlir_arm_neon_tests

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --convert-vector-to-scf --convert-scf-to-cf  --convert-vector-to-llvm='enable-arm-neon enable-arm-dotprod' \
// DEFINE:   --expand-strided-metadata --convert-to-llvm --finalize-memref-to-llvm  \
// DEFINE:   --lower-affine --convert-arith-to-llvm --reconcile-unrealized-casts \
// DEFINE: -o %t

// DEFINE: %{entry_point} = main

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void  --march=aarch64 --mattr="+neon,+dotprod" \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

// RUN: rm -f %t && %{compile} && FileCheck %s --input-file=%t -check-prefix CHECK-IR && %{run} | FileCheck %s

#mmt = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

//
// Test the lowering of `vector.contract` using the `LowerContractionToNeonSdotPattern`
//
// The operation that the `vector.contract` in this test performs is matrix
// multiplication with accumulate
//     OUT = ACC + LHS * RHS
// of two signed 8-bit integer matrices LHS and RHS, and a 32-bit integer matrix
// ACC into a 32-bit integer matrix OUT.
//
// The pattern handles input/output vectors with shapes:
//   * LHS:      vector<MxKxi8>
//   * RHS:      vector<NxKxi8>
//   * ACC, OUT: vector<MxNxi32>
// where N and K are divisible by 4. Note that the RHS is transposed.
//
// The data is the same as in the SMMLA test of vector-contract-i8mm.mlir, so the
// expected result is the same too. The vector-matrix case is covered by
// test/Dialect/ArmNeon/vector-sdot.mlir.
//

// CHECK-IR-LABEL: llvm.func @test_sdot
// CHECK-IR-COUNT-8: arm_neon.intr.sdot
func.func @test_sdot() {

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8

  // Accumulator test data
  %acc_cst = arith.constant dense<[[-44,  20,  44, -46],
                                   [ -8,  25, -34,  26],
                                   [-20, -36,  -3,  39],
                                   [-48, -31, -25, -21]]> : vector<4x4xi32>

  %acc_mem = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %acc_cst, %acc_mem[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4x4xi32>, vector<4x4xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[-35, -27, -36, -31,  23, -34,  -8, -33],
                                   [-20,  17, -32, -47,  37,  22,  -7, -21],
                                   [ -7, -35,  20,  -4,  39,  46, -23,  40],
                                   [ 40,  27,  37,  43,  38,  -6,  37,  49]]> : vector<4x8xi8>

  %lhs_mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[-17, -50,  -1,  48, -13,  22,  39,  33],
                                   [-35, -24,  37, -32,  33,  30, -11, -17],
                                   [-28,  31,   3, -44, -15, -27,  22,  35],
                                   [-23,  39,  48,  26, -23,  32, -39, -38]]> : vector<4x8xi8>

  %rhs_mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>

  // Matrix multiplication and accumulate with transposed RHS.
  %0 = arith.extsi %lhs : vector<4x8xi8> to vector<4x8xi32>
  %1 = arith.extsi %rhs : vector<4x8xi8> to vector<4x8xi32>
  %2 = vector.contract {indexing_maps = #mmt,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x8xi32>, vector<4x8xi32> into vector<4x4xi32>

  // Display the result of the multiplication
  vector.print str "Result(SDOT):\n"
  %u0 = vector.extract %2[0] : vector<4xi32> from vector<4x4xi32>
  %u1 = vector.extract %2[1] : vector<4xi32> from vector<4x4xi32>
  %u2 = vector.extract %2[2] : vector<4xi32> from vector<4x4xi32>
  %u3 = vector.extract %2[3] : vector<4xi32> from vector<4x4xi32>
  vector.print %u0 : vector<4xi32>
  vector.print %u1 : vector<4xi32>
  vector.print %u2 : vector<4xi32>
  vector.print %u3 : vector<4xi32>

  return
}

func.func @main() {
// CHECK-LABEL: Result(SDOT):
// CHECK: ( -1999,  1941,   685, -2879 )
// CHECK: ( -3705,  2952,   987,  -685 )
// CHECK: (  2565,  4157, -1589,  -357 )
// CHECK: (  2383, -2252,    32, -1365 )
  func.call @test_sdot() : () -> ()

  return
}
