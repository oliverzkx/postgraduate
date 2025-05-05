# Compiler and flags
CXX = g++
CXXFLAGS = -O2 -std=c++14

# Eigen include path (adjust if necessary)
INCLUDES = -I /usr/include/eigen3
LDFLAGS = 

# Source and object files
SRC = main.cpp poisson.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = poisson_test

# Default build rule
all: $(TARGET)

# Link object files into final binary
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile each .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

# Clean build files
clean:
	rm -f $(OBJ) $(TARGET)
