// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// SVE vectors and predicates are passed and returned in registers, as scalable
// vectors.

#include <arm_sve.h>

svint32_t add(svint32_t a, svint32_t b) { return a + b; }

// CIR: cir.func {{.*}}@add(%arg0: !cir.vector<[4] x !s32i>{{.*}}, %arg1: !cir.vector<[4] x !s32i>{{.*}}) -> !cir.vector<[4] x !s32i>
// LLVM-LABEL: define{{.*}} <vscale x 4 x i32> @add(<vscale x 4 x i32> %{{.+}}, <vscale x 4 x i32> %{{.+}})

svbool_t pred(svbool_t p, svint8_t v) { return p; }

// CIR: cir.func {{.*}}@pred(%arg0: !cir.vector<[16] x !cir.int<u, 1>>{{.*}}, %arg1: !cir.vector<[16] x !s8i>{{.*}}) -> !cir.vector<[16] x !cir.int<u, 1>>
// LLVM-LABEL: define{{.*}} <vscale x 16 x i1> @pred(<vscale x 16 x i1> %{{.+}}, <vscale x 16 x i8> %{{.+}})

float mixed(svfloat32_t v, int n) { return n; }

// CIR: cir.func {{.*}}@mixed(%arg0: !cir.vector<[4] x !cir.float>{{.*}}, %arg1: !s32i {{.*}}) -> !cir.float
// LLVM-LABEL: define{{.*}} float @mixed(<vscale x 4 x float> %{{.+}}, i32 noundef %{{.+}})

svint32_t pass(svint32_t v) { return v; }
svint32_t caller(svint32_t v) { return pass(v); }

// CIR-LABEL: cir.func {{.*}}@caller(
// CIR: cir.call @pass(%{{.+}}) : (!cir.vector<[4] x !s32i>) -> !cir.vector<[4] x !s32i>
// LLVM-LABEL: define{{.*}} <vscale x 4 x i32> @caller(
// LLVM: call <vscale x 4 x i32> @pass(<vscale x 4 x i32> %{{.+}})
