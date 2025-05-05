# 4_2_vcycle/Makefile
# ----------------------------------------------------------
CXX      := g++
CXXFLAGS := -O2 -std=c++14 -I/usr/include/eigen3      \
            -I../4_1_possion                          # reuse Poisson headers

# -----------------------------------------------------------------
# Sources that actually exist
SRC    := main.cpp mg.cpp ../4_1_possion/poisson.cpp
OBJ    := $(SRC:.cpp=.o)

TARGET := vcycle_test
# -----------------------------------------------------------------

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Generic rule – works for files in this dir *and* in ../4_1_possion
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
