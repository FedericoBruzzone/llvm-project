// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0 -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-darwin-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-darwin-cir.ll %s
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0 -emit-llvm %s -o %t-darwin.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-darwin.ll %s

// The LLVM signatures and call sites must be the same whether they come from
// ClangIR or from classic CodeGen, on AAPCS64 and on DarwinPCS.

typedef struct { int x; int y; } Pair2;
typedef struct { long a; long b; } Pair16;
typedef struct { long a, b, c, d; } Big;
typedef struct { float x, y, w, h; } HFA4;
typedef struct { double a, b; } HFA2D;
typedef struct { float a; int b; } Mixed;
typedef struct { } Empty;
typedef struct { __int128 w; } Wide;

// Scalars are passed and returned in registers.
double fp_double(double d) { return d; }

// CIR: cir.func {{.*}}@fp_double(%arg0: !cir.double {{.*}}) -> !cir.double
// LLVM-LABEL: define{{.*}} double @fp_double(double noundef %{{.+}})

// A struct of up to 8 bytes is returned in x0 as an i64.
Pair2 ret_pair2(int a) { Pair2 p = {a, a}; return p; }

// CIR: cir.func {{.*}}@ret_pair2(%arg0: !s32i {{.*}}) -> !u64i
// LLVM-LABEL: define{{.*}} i64 @ret_pair2(i32 noundef %{{.+}})

// A struct of 16 bytes is returned in x0 and x1, as an array of two i64.
Pair16 ret_pair16(long a) { Pair16 p = {a, a}; return p; }

// CIR: cir.func {{.*}}@ret_pair16(%arg0: !s64i {{.*}}) -> !cir.array<!u64i x 2>
// LLVM-LABEL: define{{.*}} [2 x i64] @ret_pair16(i64 noundef %{{.+}})

// A struct larger than 16 bytes is returned through a hidden pointer, which
// AArch64 passes in x8, so it is an sret parameter and not the first argument.
Big ret_big(long a) { Big b = {a, a, a, a}; return b; }

// CIR: cir.func {{.*}}@ret_big(%arg0: !cir.ptr<!rec_Big> {{.*}}llvm.sret = !rec_Big{{.*}}, %arg1: !s64i {{.*}})
// LLVM-LABEL: define{{.*}} void @ret_big(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{.+}}, i64 noundef %{{.+}})

// A homogeneous floating-point aggregate is returned in the vector registers.
HFA4 ret_hfa4(float a) { HFA4 h = {a, a, a, a}; return h; }

// CIR: cir.func {{.*}}@ret_hfa4(%arg0: !cir.float {{.*}}) -> !rec_HFA4
// LLVM-LABEL: define{{.*}} %struct.HFA4 @ret_hfa4(float noundef %{{.+}})

// A struct that is not homogeneous is returned as an integer.
Mixed ret_mixed(float a) { Mixed m = {a, 1}; return m; }

// CIR: cir.func {{.*}}@ret_mixed(%arg0: !cir.float {{.*}}) -> !u64i
// LLVM-LABEL: define{{.*}} i64 @ret_mixed(float noundef %{{.+}})

Wide ret_wide(long a) { Wide w = {a}; return w; }

// CIR: cir.func {{.*}}@ret_wide(%arg0: !s64i {{.*}}) -> !u128i
// LLVM-LABEL: define{{.*}} i128 @ret_wide(i64 noundef %{{.+}})

// A struct of up to 8 bytes is passed in one register.
int take_pair2(Pair2 p) { return p.x + p.y; }

// CIR: cir.func {{.*}}@take_pair2(%arg0: !u64i {{.*}}) -> !s32i
// LLVM-LABEL: define{{.*}} i32 @take_pair2(i64 %{{.+}})

// A struct of 16 bytes is passed in two registers.
long take_pair16(Pair16 p) { return p.a + p.b; }

// CIR: cir.func {{.*}}@take_pair16(%arg0: !cir.array<!u64i x 2> {{.*}}) -> !s64i
// LLVM-LABEL: define{{.*}} i64 @take_pair16([2 x i64] %{{.+}})

// A struct larger than 16 bytes is passed by reference to a caller-owned copy.
long take_big(Big b) { return b.a + b.d; }

// CIR: cir.func {{.*}}@take_big(%arg0: !cir.ptr<!rec_Big> {{.*}}llvm.dereferenceable = 32 : i64{{.*}}) -> !s64i
// LLVM-LABEL: define{{.*}} i64 @take_big(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %{{.+}})

// A homogeneous floating-point aggregate is passed as an array of its members.
float take_hfa4(HFA4 h) { return h.x + h.h; }

// CIR: cir.func {{.*}}@take_hfa4(%arg0: !cir.array<!cir.float x 4> {llvm.alignstack = 8 : i64}{{.*}}) -> !cir.float
// LLVM-LABEL: define{{.*}} float @take_hfa4([4 x float] alignstack(8) %{{.+}})

double take_hfa2d(HFA2D h) { return h.a + h.b; }

// CIR: cir.func {{.*}}@take_hfa2d(%arg0: !cir.array<!cir.double x 2> {llvm.alignstack = 8 : i64}{{.*}}) -> !cir.double
// LLVM-LABEL: define{{.*}} double @take_hfa2d([2 x double] alignstack(8) %{{.+}})

float take_mixed(Mixed m) { return m.a + m.b; }

// CIR: cir.func {{.*}}@take_mixed(%arg0: !u64i {{.*}}) -> !cir.float
// LLVM-LABEL: define{{.*}} float @take_mixed(i64 %{{.+}})

long take_wide(Wide w) { return (long)w.w; }

// CIR: cir.func {{.*}}@take_wide(%arg0: !u128i {{.*}}) -> !s64i
// LLVM-LABEL: define{{.*}} i64 @take_wide(i128 %{{.+}})

// An empty struct takes no register in C.
int take_empty(Empty e, int a) { return a; }

// CIR: cir.func {{.*}}@take_empty(%arg0: !s32i {{.*}}) -> !s32i
// LLVM-LABEL: define{{.*}} i32 @take_empty(i32 noundef %{{.+}})

// The call sites follow the same classification as the definitions.
long call_big(long a) {
  Big b = ret_big(a);
  return take_big(b);
}

// CIR-LABEL: cir.func {{.*}}@call_big(
// CIR: cir.call @ret_big(%{{.+}}, %{{.+}}) : (!cir.ptr<!rec_Big> {{.*}}llvm.sret = !rec_Big{{.*}}, !s64i {llvm.noundef}) -> ()
// CIR: cir.call @take_big(%{{.+}}) : (!cir.ptr<!rec_Big> {{.*}}llvm.dereferenceable = 32 : i64{{.*}}) -> !s64i
// LLVM-LABEL: define{{.*}} i64 @call_big(
// LLVM: call void @ret_big(ptr dead_on_unwind writable sret(%struct.Big) align 8 %{{.+}}, i64 noundef %{{.+}})
// LLVM: call i64 @take_big(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %{{.+}})

float call_hfa(float a) {
  HFA4 h = ret_hfa4(a);
  return take_hfa4(h);
}

// CIR-LABEL: cir.func {{.*}}@call_hfa(
// CIR: cir.call @ret_hfa4(%{{.+}}) : (!cir.float {llvm.noundef}) -> !rec_HFA4
// CIR: cir.call @take_hfa4(%{{.+}}) : (!cir.array<!cir.float x 4> {llvm.alignstack = 8 : i64}) -> !cir.float
// LLVM-LABEL: define{{.*}} float @call_hfa(
// LLVM: call %struct.HFA4 @ret_hfa4(float noundef %{{.+}})
// LLVM: call float @take_hfa4([4 x float] alignstack(8) %{{.+}})
