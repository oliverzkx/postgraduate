# Makefile for Case_3
# Adjust CXXFLAGS as needed, e.g., -Wall -std=c++11, etc.

CXX      = g++
CXXFLAGS = -O2 -Wall
TARGETS  = run_3_1 run_3_2 run_3_3

all: $(TARGETS)

# Build run_3_1 from 3_1.cc
run_3_1: 3_1.cc
	$(CXX) $(CXXFLAGS) 3_1.cc -o run_3_1

# Build run_3_2 from 3_2.cc
run_3_2: 3_2.cc
	$(CXX) $(CXXFLAGS) 3_2.cc -o run_3_2

# Build run_3_3 from 3_3.cc
run_3_3: 3_3.cc
	$(CXX) $(CXXFLAGS) 3_3.cc -o run_3_3

# 'clean' target to remove the executables, CSV, or PNG files if desired
clean:
	rm -f $(TARGETS) *.csv *.png
