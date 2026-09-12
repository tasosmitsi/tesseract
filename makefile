# ============================================================================
# tesseract build
#
#   make                           release: libtesseract.a + tests + example
#   make MODE=debug                -Og -g3
#   make MODE=sanitize             ASan + UBSan
#   make TRACY=1                   Tracy client, in its own build tree
#   make CXX=clang++ CC=clang      clang toolchain (auto-detected; llvm-ar, -flto)
#   make CXXSTD=c++20              language standard (also CSTD, THIRD_PARTY_STD)
#   make ARCH=                     portable build (default: -march=native)
#   make DEFS=-DTESSERACT_ARM_UARCH_A76      uarch defines (A76 / A72 / A55)
#   make run_test ARGS='[cholesky]'
#   make freestanding              syntax-check every header with no libc++, no
#                                  exceptions, __STDC_HOSTED__ == 0
#   make compile_commands.json     clangd database (needs bear)
#   make print-CXXFLAGS            print any variable
#   make clean                     drop this build tree's objects (keeps Catch2)
#   make clean_all                 remove build/ entirely
#
# Objects mirror the source tree under build/<MODE>/, so identical filenames in
# core/src/, tests/ and examples/src/ never collide. MODE, TRACY, ARCH, DEFS and
# the standard versions all change code generation, so MODE and TRACY get their
# own tree and the rest go into a stamp that forces a rebuild when they move.
# ============================================================================

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.SUFFIXES:
MAKEFLAGS += --no-builtin-rules

# ---- Toolchain --------------------------------------------------------------
CXX = g++-14
CC  = gcc-14

# Clang is detected from whatever CXX resolves to, so a single
#   make CXX=clang++ CC=clang
# flips the flag set and the archiver in one move. Override both, or C files
# get gcc with clang-only flags. `make print-CXX_IS_CLANG` to check.
CXX_IS_CLANG := $(shell $(CXX) --version 2>/dev/null | grep -qi clang && echo 1 || echo 0)

ifeq ($(CXX_IS_CLANG),1)
  # Plain ar can't build a symbol index over LLVM bitcode members (-flto).
  AR = llvm-ar
else
  # Switch to gcc-ar-14 if you ever enable -flto on gcc.
  AR = ar
endif

# ---- Language standards -----------------------------------------------------
# Named once, used by the compile rules, the third-party rules and the
# freestanding check. Override to test portability:
#   make CXXSTD=c++20 freestanding
# Third-party sources get their own, since Catch2 and Tracy have no reason to
# track whatever tesseract is on.
CXXSTD          ?= c++23
CSTD            ?= c17
THIRD_PARTY_STD ?= c++23

# ---- Directories ------------------------------------------------------------
CORE_SRC_DIR    = core/src
CORE_INC_DIR    = core/include
TEST_DIR        = tests/catch2_tests
EXAMPLE_SRC_DIR = examples/src
EXAMPLE_INC_DIR = examples/include
CATCH2_DIR      = Catch2/extras
TRACY_DIR       = third_party/tracy

BUILD_ROOT = build
MODE      ?= release
TRACY     ?= 0

# A TRACY=1 build instruments every zone, which contaminates Catch2's absolute
# timings. Separate trees let the clean build and the profiling build coexist
# instead of overwriting each other's objects.
BUILD_DIR = $(BUILD_ROOT)/$(MODE)$(if $(filter-out 0,$(TRACY)),-tracy)

# ---- Common flags -----------------------------------------------------------
WARNINGS = -Wall -Wextra -Wpedantic -Wshadow
# Worthwhile on index-arithmetic-heavy templates, but budget a triage session:
# WARNINGS += -Wconversion

DEPFLAGS = -MMD -MP

# Extra defines, e.g. make DEFS=-DTESSERACT_ARM_UARCH_A76 (also: A72, A55)
DEFS ?=

# Applies to every mode. The microkernels pick their path from -march, so a
# debug or sanitize build without it exercises different code than ships.
ARCH ?= -march=native

# ---- Build modes ------------------------------------------------------------
# Inlining is tesseract's whole performance story (expression templates and SIMD
# microkernels), so release raises the inliner limits and adds -Winline to hear
# about anything that refused. Debug and sanitize skip both: at -Og the inliner
# barely runs and -Winline would only be noise.
ifeq ($(MODE),release)
  ifeq ($(CXX_IS_CLANG),1)
    # With -flto codegen happens at link time, so the link line carries the same
    # optimization flags as the compile lines.
    MODE_FLAGS   = -O3 -flto -mllvm -inline-threshold=500000
    MODE_LDFLAGS = $(MODE_FLAGS) $(ARCH)
    WARNINGS    += -Winline -Rpass-missed=inline
  else
    MODE_FLAGS   = -O3 --param large-function-growth=1500 --param inline-unit-growth=300
    MODE_LDFLAGS =
    WARNINGS    += -Winline
  endif
else ifeq ($(MODE),debug)
  MODE_FLAGS   = -Og -g3
  MODE_LDFLAGS =
else ifeq ($(MODE),sanitize)
  SAN          = -fsanitize=address,undefined
  MODE_FLAGS   = -Og -g3 $(SAN) -fno-omit-frame-pointer
  MODE_LDFLAGS = $(SAN)
else
  $(error unknown MODE '$(MODE)': use release, debug or sanitize)
endif

# ---- Third party ------------------------------------------------------------
# Sources say #include <Dense>. Point this at the parent to use <Eigen/Dense>.
EIGEN_INC = /usr/local/src/eigen/Eigen

# python3-config tracks whatever Python is installed instead of hardcoding 3.12.
# Both expand empty if python3-dev is absent, and the build still works minus
# the Python-dependent tests.
PY_INC  := $(shell python3-config --includes 2>/dev/null)
PY_LIBS := $(shell python3-config --ldflags --embed 2>/dev/null)

ifneq ($(filter-out 0,$(TRACY)),)
  DEFS     += -DTRACY_ENABLE
  TRACY_INC = -isystem$(TRACY_DIR)/public
  TRACY_OBJ = $(BUILD_DIR)/third_party/TracyClient.o
  LDLIBS   += -ldl -lpthread
endif

# ---- Composed flags ---------------------------------------------------------
INCLUDES = -I$(CORE_INC_DIR) -I$(EXAMPLE_INC_DIR) $(TRACY_INC)

CXXFLAGS = -std=$(CXXSTD) $(WARNINGS) $(MODE_FLAGS) $(ARCH) $(DEFS) $(INCLUDES) $(DEPFLAGS)
CFLAGS   = -std=$(CSTD)   $(WARNINGS) $(MODE_FLAGS) $(ARCH) $(DEFS) $(INCLUDES) $(DEPFLAGS)
LDFLAGS  = $(MODE_LDFLAGS)

# Catch2, Eigen and Python are test-only. Eigen is the cross-check reference and
# Python drives the plotting; neither belongs in the search path of a library
# that claims to be STL-free. /usr/include/pythonX.Y is the loudest of the three,
# full of generic names (object.h, token.h, ast.h) inviting collisions.
TEST_ONLY_INC = -isystem$(CATCH2_DIR) -isystem$(EIGEN_INC) $(PY_INC)
$(BUILD_DIR)/$(TEST_DIR)/%.o: private CXXFLAGS += $(TEST_ONLY_INC)
$(BUILD_DIR)/$(TEST_DIR)/%.o: private CFLAGS   += $(TEST_ONLY_INC)

# ---- Rebuild on flag change -------------------------------------------------
# MODE and TRACY get their own tree, but ARCH, DEFS and the standards don't, and
# stale objects built for another -march link without complaint. Stamp the
# flags; every object depends on the stamp.
#
# CXX and CC lead the string because the flags alone can't tell the toolchains
# apart: in debug and sanitize both compilers get identical MODE_FLAGS, so
# switching to clang would otherwise reuse gcc's objects.
FLAGS_STAMP = $(BUILD_DIR)/.flags
FLAGS_NOW   = $(CXX) | $(CC) | $(CXXFLAGS) | $(CFLAGS) | $(LDFLAGS) | $(LDLIBS) | $(THIRD_PARTY_STD)
$(shell mkdir -p $(BUILD_DIR); \
        printf '%s' '$(FLAGS_NOW)' | cmp -s - $(FLAGS_STAMP) \
        || printf '%s' '$(FLAGS_NOW)' > $(FLAGS_STAMP))

# ---- Objects: build tree mirrors source tree --------------------------------
# $(call objs,dir) -> one object path per .cpp and .c under dir, at any depth.
# sort makes the order stable, so link lines don't shuffle between machines.
objs = $(patsubst %,$(BUILD_DIR)/%.o,$(basename $(sort \
         $(shell find $1 \( -name '*.cpp' -o -name '*.c' \) 2>/dev/null))))

CORE_OBJS    = $(call objs,$(CORE_SRC_DIR))
TEST_OBJS    = $(call objs,$(TEST_DIR))
EXAMPLE_OBJS = $(call objs,$(EXAMPLE_SRC_DIR))
CATCH_OBJ    = $(BUILD_DIR)/third_party/catch_amalgamated.o

ALL_OBJS = $(CORE_OBJS) $(TEST_OBJS) $(EXAMPLE_OBJS) $(CATCH_OBJ) $(TRACY_OBJ)
$(ALL_OBJS): $(FLAGS_STAMP)

# ---- Targets ----------------------------------------------------------------
LIB            = $(BUILD_DIR)/libtesseract.a
TEST_TARGET    = $(BUILD_DIR)/test
EXAMPLE_TARGET = $(BUILD_DIR)/example

# Build only what has sources: an empty core/src, tests/ or examples/src is
# skipped rather than producing an empty archive or a link error.
TARGETS =
ifneq ($(strip $(CORE_OBJS)),)
  TARGETS += $(LIB)
  LINK_LIB = $(LIB)
endif
ifneq ($(strip $(TEST_OBJS)),)
  TARGETS += $(TEST_TARGET)
endif
ifneq ($(strip $(EXAMPLE_OBJS)),)
  TARGETS += $(EXAMPLE_TARGET)
endif

all: $(TARGETS)

# The archive goes last on every link line.
$(LIB): $(CORE_OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

$(TEST_TARGET): $(TEST_OBJS) $(CATCH_OBJ) $(TRACY_OBJ) $(LINK_LIB)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS) $(PY_LIBS)

$(EXAMPLE_TARGET): $(EXAMPLE_OBJS) $(TRACY_OBJ) $(LINK_LIB)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# ---- Compile rules: two patterns cover every source dir ---------------------
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

# Third-party sources: own subdir, warnings off via -w, still on MODE_FLAGS so a
# sanitize build is instrumented end to end. -Winline and -Rpass live in
# WARNINGS rather than MODE_FLAGS, so neither emits inline diagnostics. These
# explicit rules beat the patterns above.
$(CATCH_OBJ): $(CATCH2_DIR)/catch_amalgamated.cpp
	@mkdir -p $(@D)
	$(CXX) -std=$(THIRD_PARTY_STD) -w $(MODE_FLAGS) $(ARCH) -isystem$(CATCH2_DIR) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR)/third_party/TracyClient.o: $(TRACY_DIR)/public/TracyClient.cpp
	@mkdir -p $(@D)
	$(CXX) -std=$(THIRD_PARTY_STD) -w $(MODE_FLAGS) $(ARCH) $(DEFS) $(TRACY_INC) $(DEPFLAGS) -c $< -o $@

# ---- Checks -----------------------------------------------------------------
# Every header, compiled with no libstdc++ headers, no exceptions, no RTTI and
# __STDC_HOSTED__ == 0. This is the STL-free claim as a build step instead of a
# convention: an accidental #include <type_traits> fails here and nowhere else.
FREESTANDING_TU    = $(BUILD_DIR)/freestanding.cpp
FREESTANDING_FLAGS = -std=$(CXXSTD) -fsyntax-only -ffreestanding -fno-exceptions \
                     -fno-rtti -nostdinc++ $(WARNINGS) $(ARCH) $(DEFS) $(INCLUDES)

freestanding:
	@mkdir -p $(BUILD_DIR)
	@find $(CORE_INC_DIR) \( -name '*.h' -o -name '*.hpp' \) | sort \
	  | sed 's|^$(CORE_INC_DIR)/|#include "|; s|$$|"|' > $(FREESTANDING_TU)
	$(CXX) $(FREESTANDING_FLAGS) $(FREESTANDING_TU)
	@echo "headers are freestanding-clean"

# ---- Convenience ------------------------------------------------------------
run_test: $(TEST_TARGET)
	./$(TEST_TARGET) $(ARGS)

run_example: $(EXAMPLE_TARGET)
	./$(EXAMPLE_TARGET) $(ARGS)

build_core: $(LIB)
	@echo "Core library built: $(LIB)"

build_tests: $(TEST_TARGET)
	@echo "Test program built: $(TEST_TARGET)"

build_example: $(EXAMPLE_TARGET)
	@echo "Example program built: $(EXAMPLE_TARGET)"

# clangd on a template-heavy codebase is worth the bear dependency.
compile_commands.json:
	bear -- $(MAKE) -B $(TARGETS)

# make print-CXXFLAGS, make print-TEST_OBJS
print-%:
	@echo '$* = $($*)'

# Everything in this tree except the third-party objects, which cost minutes to
# rebuild and never change.
clean:
	rm -rf $(filter-out $(BUILD_DIR)/third_party,$(wildcard $(BUILD_DIR)/*))

clean_all:
	rm -rf $(BUILD_ROOT)

.PHONY: all clean clean_all run_test run_example build_core build_tests \
        build_example freestanding compile_commands.json

# Dependency includes stay at the bottom: included files contribute targets, and
# any target parsed before `all` would hijack the default goal. .DEFAULT_GOAL at
# the top makes this double-safe.
-include $(ALL_OBJS:.o=.d)
