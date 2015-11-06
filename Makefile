MMAP_FILES ?= ./tagged
MP_DATA ?= ./mp-data

RAW = $(MMAP_FILES)/train.labels.db \
      $(MMAP_FILES)/train.images.db \
      $(MMAP_FILES)/val.images.db \
      $(MMAP_FILES)/test.images.db \

solve: $(RAW)
	./main.py $(MMAP_FILES)

$(RAW): prepare.py
	mkdir -p $(MMAP_FILES)
	./prepare.py $(MP_DATA)/images/ $(MMAP_FILES)

clean:
	rm -f $(RAW)
