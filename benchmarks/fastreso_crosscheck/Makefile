FASTRESO_DIR ?= ../../../FastReso
FASTRESO_BUILD ?= $(FASTRESO_DIR)/build
GSL_PREFIX ?=

CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17
OPENMP_FLAG ?= -fopenmp
CPPFLAGS += -I$(FASTRESO_DIR)
LDFLAGS +=
LDLIBS += $(FASTRESO_BUILD)/libfastreso.a -lgsl -lgslcblas -lm

ifneq ($(strip $(GSL_PREFIX)),)
CPPFLAGS += -I$(GSL_PREFIX)/include
LDFLAGS += -L$(GSL_PREFIX)/lib
endif

TARGET := bin/fastreso_reference
SOURCE := fastreso_reference.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE) $(FASTRESO_BUILD)/libfastreso.a
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAG) $< \
	    $(LDFLAGS) $(LDLIBS) -o $@

clean:
	$(RM) $(TARGET)
