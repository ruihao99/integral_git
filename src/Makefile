#=======================================================================
#                   Compiler and Toolchain
#=======================================================================

CC          = gcc
F90         = gfortran
PYTHON      = python
F90WRAP     = f90wrap
F2PY        = f2py-f90wrap

#=======================================================================
#                   Platform-dependent settings
#=======================================================================

UNAME_S     := $(shell uname -s)
EXT         = so
ifeq ($(OS),Windows_NT)
    EXT     = pyd
endif

#=======================================================================
#                   Project-specific variables
#=======================================================================

PY_MOD      = mori_gqme
F90_SRC     = mod_mori_gqme.f90
OBJ         = $(F90_SRC:.f90=.o)
F90WRAP_SRC = $(addprefix f90wrap_,${F90_SRC})
WRAPFLAGS   = -v --type-check --kind-map kind_map
F2PYFLAGS   = --build-dir build
TARGET      = _${PY_MOD}.${EXT}

#=======================================================================
#                   Build rules
#=======================================================================

.PHONY: all clean f2py test

all: test

# Clean all generated files
clean:
	rm -rf *.mod *.smod *.o f90wrap_*.f90 ${PY_MOD}.py _${PY_MOD}.* __pycache__/ .f2py_f2cmap build ${PY_MOD}/

# Compile Fortran source
%.o: %.f90
	$(F90) -c -fPIC -O2 $< -o $@

# Generate f90wrap file from Fortran module
${F90WRAP_SRC}: ${OBJ}
	$(F90WRAP) -m ${PY_MOD} ${WRAPFLAGS} ${F90_SRC}

# Build Python extension using f2py and gfortran
f2py: ${F90WRAP_SRC} ${OBJ}
	CFLAGS="-fPIC" $(F2PY) -c -m _${PY_MOD} ${F2PYFLAGS} f90wrap_*.f90 ${OBJ}

# Run the test script after building the extension
test: f2py
	$(PYTHON) test.py

