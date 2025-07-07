#=======================================================================
#                   define the compiler names
#=======================================================================

CC          = gcc
F90         = gfortran
PYTHON      = python
UNAME_S     := $(shell uname -s)
EXT         = so

ifeq ($(OS),Windows_NT)
    EXT     = pyd
endif

CFLAGS      = -fPIC
F90FLAGS    = -fPIC -O2
PY_MOD      = mori_gqme
F90_SRC     = mod_mori_gqme.f90
OBJ         = $(F90_SRC:.f90=.o)
F90WRAP_SRC = $(addprefix f90wrap_,${F90_SRC})
WRAPFLAGS   = -v --type-check --kind-map kind_map
F2PYFLAGS   = --build-dir build
F90WRAP     = f90wrap
F2PY        = f2py-f90wrap

TARGET      = _${PY_MOD}.$(EXT)

.PHONY: all clean

all: test

clean:
	rm -rf *.mod *.smod *.o f90wrap_*.f90 ${PY_MOD}.py _${PY_MOD}.* __pycache__/ .f2py_f2cmap build ${PY_MOD}/

# main.o: ${F90_SRC}
# 	${F90} ${F90FLAGS} -c $< -o $@

%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

${F90WRAP_SRC}: ${OBJ}
	${F90WRAP} -m ${PY_MOD} ${WRAPFLAGS} ${F90_SRC}

$(TARGET): ${F90WRAP_SRC}
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD} ${F2PYFLAGS} f90wrap_*.f90 *.o

f2py: $(TARGET)

test: f2py
	${PYTHON} test.py
