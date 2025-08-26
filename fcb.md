cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="" -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Release
ninja -C build
build/bin/opt -disable-output /tmp/a.ll -passes=fcb


// LLVM TEST SUITE

 cmake -B build \
      -DCMAKE_C_COMPILER=/Users/federicobruzzone/dev/llvm-project/build/bin/clang \
      -DCMAKE_CXX_COMPILER=/Users/federicobruzzone/dev/llvm-project/build/bin/clang++ \
      -DCMAKE_C_FLAGS="-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" \
      -DCMAKE_CXX_FLAGS="-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" \
      -DTEST_SUITE_BENCHMARKING_ONLY=ON \
      -DTEST_SUITE_RUN_BENCHMARKS=ON \
      -C./cmake/caches/O3.cmake

// from the build directory, run:
make -j16


/Users/federicobruzzone/dev/llvm-project/build/bin/llvm-lit -v -j 1 -o results.json .

-DDEFAULT_SYSROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" \





// ===================
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS=all [-DLLVM_ENABLE_RUNTIMES=all] -DCMAKE_BUILD_TYPE=Debug
ninja -C build [llvm-project] // build all LLVM projects

ninja -C build check-llvm // run all tests
ninja -C build check-clang // run all tests
grep -A 20 "LLVM_ENABLE_PROJECTS" llvm/CMakeLists.txt | grep "polly"
