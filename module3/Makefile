NVCC ?= nvcc
TARGET := assignment.exe
SOURCE := assignment.cu
NVCCFLAGS := -O2 -std=c++17

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)
