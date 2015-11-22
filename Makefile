MMAP_FILES ?= ./tagged
DK_DATA ?= ./mp-dev_kit
MP_DATA ?= ./mp-data
PYTHON = env/bin/python3

RAW = $(MMAP_FILES)/full/train.labels.db \
      $(MMAP_FILES)/full/train.images.db \
      #$(MMAP_FILES)/full/val.images.db \
      #$(MMAP_FILES)/full/test.images.db \

SRAW = $(MMAP_FILES)/small/train.labels.db \
       $(MMAP_FILES)/small/train.images.db \
       #$(MMAP_FILES)/small/val.images.db \
       #$(MMAP_FILES)/small/test.images.db \

IMDATA = $(MP_DATA)/images/train/y/yard/00001000.jpg

all: $(IMDATA) solve

$(PYTHON) env: env.sh
	sh env.sh

$(IMDATA):
	mkdir -p $(MP_DATA)
	curl "http://6.869.csail.mit.edu/fa15/challenge/data.tar.gz" -o $(MP_DATA)/data.tar.gz
	tar mxvzf $(MP_DATA)/data.tar.gz -C $(MP_DATA)

solve-small: $(PYTHON) $(SRAW) Makefile
	$(PYTHON) main.py -p plot.png -c network.mdl -e5 -b30 -s5 $(MMAP_FILES)/small

solve: $(PYTHON) $(RAW) Makefile
	$(PYTHON) main.py -l test-labels.db -p plot-large.png -c network-large.mdl $(MMAP_FILES)/full

$(SRAW): $(PYTHON) $(IMDATA) prepare.py
	mkdir -p $(MMAP_FILES)/small
	$(PYTHON) prepare.py -c10 -s200 $(MP_DATA)/images/ $(DK_DATA) $(MMAP_FILES)/small

$(RAW): $(PYTHON) $(IMDATA) prepare.py
	mkdir -p $(MMAP_FILES)/full
	$(PYTHON) prepare.py $(MP_DATA)/images/ $(DK_DATA) $(MMAP_FILES)/full

clean:
	rm -f $(RAW) $(SRAW)
