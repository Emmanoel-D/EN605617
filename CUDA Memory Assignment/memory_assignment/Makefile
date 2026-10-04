NVCC ?= nvcc
TARGET := assignment.exe
SOURCE := assignment.cu
NVCCFLAGS := -O2 -std=c++17 -arch=native -Xptxas -v

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET) run_output.txt
