CXX := g++
CXXFLAGS := -Wall -Wextra -Isrc -lOpenCL

SRCDIR := src
BINDIR := bin

SRCS := $(shell find $(SRCDIR) -name '*.cpp')

NAME := Main

DEFS := -DCL_TARGET_OPENCL_VERSION=100

all:
	$(CXX) -Ofast -o $(BINDIR)/$(NAME) $(SRCS) $(CXXFLAGS) $(DEFS) -DNDEBUG

debug:
	$(CXX) -o $(BINDIR)/$(NAME)Debug $(SRCS) $(CXXFLAGS) $(DEFS)