//===- CIRABIClassifierAArch64.cpp - AArch64 classification ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64 classifier, which classifies with the LLVM
// ABI Lowering Library's AArch64 target.
//
//===----------------------------------------------------------------------===//

#include "CIRABIClassifier.h"
#include "TargetLowering/LowerModule.h"

#include "clang/Basic/LangOptions.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/TargetParser/Triple.h"

using namespace mlir;
using namespace mlir::abi;
using namespace cir;

namespace {

class AArch64ABIClassifier final : public CIRABIClassifierCommon {
public:
  AArch64ABIClassifier(const mlir::DataLayout &dl, mlir::ModuleOp modOp)
      : CIRABIClassifierCommon(dl, modOp) {}

  mlir::FailureOr<ArgClassification>
  classifyVAArg(cir::VAArgOp vaArg) override {
    vaArg->emitOpError()
        << "not yet implemented in CallConvLowering for AArch64";
    return mlir::failure();
  }

protected:
  /// AArch64 has no per-function feature that changes its ABI rules.
  const llvm::abi::TargetInfo &getTargetInfo(cir::FuncOp) override;

  /// SVE vectors and predicates are passed in registers.
  bool allowsScalableVectors() const override { return true; }

private:
  std::unique_ptr<llvm::abi::TargetInfo> target;
};

const llvm::abi::TargetInfo &AArch64ABIClassifier::getTargetInfo(cir::FuncOp) {
  if (!target) {
    std::unique_ptr<cir::LowerModule> lowerModule =
        cir::createLowerModule(modOp);
    assert(lowerModule && "LowerModule shouldn't be null");
    const clang::LangOptions &langOpts = lowerModule->getLangOpts();

    // The same flags classic CodeGen derives in
    // CodeGenModule::getLLVMABITargetInfo.  Without cir.triple there is no
    // target, so the options stay at AAPCS on a little endian target.
    llvm::abi::AArch64ABIOptions opts;
    if (lowerModule->hasTarget()) {
      const clang::TargetInfo &targetInfo = lowerModule->getTarget();
      const llvm::Triple &triple = targetInfo.getTriple();
      llvm::StringRef abi = targetInfo.getABI();
      if (abi == "darwinpcs")
        opts.Kind = llvm::abi::AArch64ABIKind::DarwinPCS;
      else if (triple.isOSWindows())
        opts.Kind = llvm::abi::AArch64ABIKind::Win64;
      else if (abi == "aapcs-soft")
        opts.Kind = llvm::abi::AArch64ABIKind::AAPCSSoft;
      opts.IsILP32 = triple.getArch() == llvm::Triple::aarch64_32;
      opts.IsMachO = triple.isOSBinFormatMachO();
      opts.IsAndroidOrOHOS = triple.isAndroid() || triple.isOHOSFamily();
      opts.IsWindowsArm64EC = triple.isWindowsArm64EC();
      opts.IsMicrosoftCXXABI = targetInfo.getCXXABI().isMicrosoft();
      opts.IsBigEndian = !triple.isLittleEndian();
    }
    opts.IsCXX = langOpts.CPlusPlus;
    opts.CompatInfo.IsMatrixHA =
        langOpts.getClangABICompat() > clang::LangOptions::ClangABI::Ver23;

    target =
        llvm::abi::createAArch64TargetInfo(typeMapper.getTypeBuilder(), opts);
  }
  return *target;
}

} // namespace

std::unique_ptr<CIRABIClassifier>
cir::createAArch64ABIClassifier(const mlir::DataLayout &dl,
                                mlir::ModuleOp modOp) {
  return std::make_unique<AArch64ABIClassifier>(dl, modOp);
}
