//===-- Fcb.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Fcb.h"
#include "llvm/IR/Module.h"

using namespace llvm;

PreservedAnalyses FcbPass::run(Module &M, ModuleAnalysisManager &AM) {
  errs() << "Module Name: " << M.getName() << "\n";
  return PreservedAnalyses::all();
}
