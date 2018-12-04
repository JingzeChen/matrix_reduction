CXX				=	nvcc
DOXYGEN			=	doxygen
CXX_SRC			=	$(wildcard *.cu)
DOC_SRC			=	./Doxyfile
CXX_CFLAG		=	-dc -std=c++14 -ccbin=g++-6 -O3 -use_fast_math -m64 --gpu-architecture=compute_61 --gpu-code=compute_61
CXX_LFLAG		=	--gpu-architecture=compute_61 --gpu-code=compute_61
CXX_INC_DIR		=	-Iphat_lib -Imallocmc_lib
CXX_OBJ			=	$(patsubst %.cu, %.o, $(CXX_SRC))
CXX_EXE			=	cuda_phat

.PHONY: all clean doc

all : $(CXX_OBJ)
	$(CXX) $(CXX_LFLAG) $(CXX_OBJ) -o $(CXX_EXE)

%.o : %.cu
	$(CXX) $(CXX_CFLAG) $(CXX_INC_DIR) $< -o $@

doc	:
	$(DOXYGEN) $(DOC_SRC)
	cp test_result.* documentation

clean:
	rm -rf *.o $(CXX_EXE)
