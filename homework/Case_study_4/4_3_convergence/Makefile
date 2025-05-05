CXX      := g++
CXXFLAGS := -O2 -std=c++14 -I/usr/include/eigen3

SRC := main.cpp poisson.cpp mg.cpp
OBJ := $(SRC:.cpp=.o)
TARGET := convergence_runner

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
