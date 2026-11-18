// RUN: not %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s

// cir.va_arg on AArch64 is not expanded yet, and the generic LLVM va_arg
// instruction does not follow the AAPCS64 va_list, so it is rejected.

int sum(int n, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, n);
  int res = __builtin_va_arg(args, int);
  __builtin_va_end(args);
  return res;
}

// CHECK: error: 'cir.va_arg' op not yet implemented in CallConvLowering for AArch64
