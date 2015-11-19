MMAP_FILES ?= ./tagged
DK_DATA ?= ./mp-dev_kit
MP_DATA ?= ./mp-data

RAW = $(MMAP_FILES)/full/train.labels.db \
      $(MMAP_FILES)/full/train.images.db \
      #$(MMAP_FILES)/full/val.images.db \
      #$(MMAP_FILES)/full/test.images.db \

SRAW = $(MMAP_FILES)/small/train.labels.db \
       $(MMAP_FILES)/small/train.images.db \
       #$(MMAP_FILES)/small/val.images.db \
       #$(MMAP_FILES)/small/test.images.db \

solve-small: $(SRAW) Makefile
	./main.py -p plot.png -c network.mdl -e5 -b30 -s5 $(MMAP_FILES)/small

solve: $(RAW) Makefile
	./main.py -l test-labels.db -p plot-large.png -c network-large.mdl $(MMAP_FILES)/full

$(SRAW): prepare.py Makefile
	mkdir -p $(MMAP_FILES)/small
	./prepare.py -c10 -s200 $(MP_DATA)/images/ $(DK_DATA) $(MMAP_FILES)/small

$(RAW): prepare.py Makefile
	mkdir -p $(MMAP_FILES)/full
	./prepare.py $(MP_DATA)/images/ $(DK_DATA) $(MMAP_FILES)/full

clean:
	rm -f $(RAW) $(SRAW)
