CXX := g++
CXXFLAGS := -Wall -Wextra -Isrc -lOpenCL

BINDIR := bin
SRCDIR := src

NAME := Main
OUT := -o $(BINDIR)/$(NAME)
SRCS := $(shell find $(SRCDIR) -name '*.cpp')
DEFS := -DCL_TARGET_OPENCL_VERSION=100

all:
	$(CXX) -Ofast $(OUT) $(SRCS) $(CXXFLAGS) $(DEFS) -DNDEBUG

debug:
	$(CXX) $(OUT)Debug $(SRCS) $(CXXFLAGS) $(DEFS)