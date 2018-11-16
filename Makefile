CXX = nvcc
SRC =  gpu_spectral_sequence_reduction.cu gpu_common.cu chunk_reduction_algorithm.cu gpu_boundary_matrix.cu
OBJ = $(patsubst %.cu,%.o,${SRC})
OBJ_DEBUG = $(patsubst %.cu,%_debug.o,${SRC})

CXXFLAGS_COMMON = -std=c++14 -Iphat_lib -Imallocmc_lib
CXXFLAGS_DEBUG =  ${CXXFLAGS_COMMON} -g -G -O0
CXXFLAGS_RELEASE = ${CXXFLAGS_COMMON} -O3 -DBDEBUG

TARGET_DEBUG = matReduct_debug
TARGET = matReduct

LDFLAGS = -lcuda

.PHONY: debug release clean

debug: ${TARGET_DEBUG}

release: ${TARGET}

clean:
	rm -rf ${TARGET}
	rm -rf ${TARGET_DEBUG}
	rm -rf ${OBJ}
	rm -rf ${OBJ_DEBUG}

${TARGET_DEBUG} : ${OBJ_DEBUG}
	$(CXX) ${CXXFLAGS_DEBUG} $^ -o $@ ${LDFLAGS}

${TARGET} : ${OBJ}
	$(CXX) ${CXXFLAGS_RELEASE} $^ -o $@ ${LDFLAGS}

%_debug.o: %.cu
	$(CXX) ${CXXFLAGS_DEBUG} -dc $^ -o $@ ${LDFLAGS}

%.o: %.cu
	$(CXX) ${CXXFLAGS} -dc $^ -o $@ ${LDFLAGS}

