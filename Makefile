MMAP_FILES ?= ./tagged
DK_DATA ?= ./mp-dev_kit
MP_DATA ?= ./mp-data
VENV = env/.built
PYTHON = env/bin/python3
NET ?= slim
LIMIT ?= 0
FOCUS ?= /home/davidbau/public/exp-deeper/resp

RAW = $(MMAP_FILES)/full/train.labels.db \
      $(MMAP_FILES)/full/train.images.db \
      $(MMAP_FILES)/full/train.filenames.txt \
      #$(MMAP_FILES)/full/val.images.db \
      #$(MMAP_FILES)/full/test.images.db

NGRAW = $(MMAP_FILES)/focus-not-goal/train.labels.db \
        $(MMAP_FILES)/focus-not-goal/train.images.db \
        $(MMAP_FILES)/focus-not-goal/train.filenames.txt

GRAW = $(MMAP_FILES)/focus-goal/train.labels.db \
       $(MMAP_FILES)/focus-goal/train.images.db \
       $(MMAP_FILES)/focus-goal/train.filenames.txt

SRAW = $(MMAP_FILES)/small/train.labels.db \
       $(MMAP_FILES)/small/train.images.db \
       $(MMAP_FILES)/small/train.filenames.txt \
       #$(MMAP_FILES)/small/val.images.db \
       #$(MMAP_FILES)/small/test.images.db

LAST = train/y/yard/00001000.jpg
IMTGZ = $(MMAP_FILES)/data.tar.gz
IMDATA = $(MP_DATA)/images/$(LAST)

all: $(IMDATA) solve

$(VENV) env: env.sh
	sh env.sh
	touch $(VENV)

$(IMTGZ):
	mkdir -p $(MP_DATA)
	curl "http://6.869.csail.mit.edu/fa15/challenge/data.tar.gz" -o $@
	touch -d '2015-09-01' $@ # avoid rebuilds

.PRECIOUS: $(IMTGZ) $(RAW) $(SRAW) $(GRAW) $(BRAW)

$(IMDATA): $(IMTGZ)
	tar mxvzf $< -C $(MP_DATA)

solve-small: $(VENV) $(SRAW) Makefile
	$(PYTHON) main.py --network $(NET) -e1 -b10 -s1 --tagged $(MMAP_FILES)/small

solve: $(VENV) $(RAW) Makefile
	$(PYTHON) main.py \
                --network $(NET) \
		--tagged $(MMAP_FILES)/full

analyze: $(VENV) $(RAW) Makefile
	$(PYTHON) main.py \
                --network $(NET) \
		--confusion \
		--response \
                --limit $(LIMIT) \
		--tagged $(MMAP_FILES)/full


view: $(VENV)
	$(PYTHON) view.py \
                --tagged $(MMAP_FILES)/full \
                -d $(DK_DATA) \
                --serve \
                --network $(NET)

# these technically depend on $(PYTHON), but we don't want to add that
# dependency, because then we have to re-prepare if we ever change env.sh
$(SRAW): $(IMDATA) prepare.py
	mkdir -p $(MMAP_FILES)/small
	$(PYTHON) prepare.py -c10 -s200 $(MP_DATA)/images/ $(DK_DATA) $(MMAP_FILES)/small

$(RAW): $(IMDATA) prepare.py
	mkdir -p $(MMAP_FILES)/full
	$(PYTHON) prepare.py $(MP_DATA)/images/ $(DK_DATA) $(MMAP_FILES)/full

$(GRAW): $(FOCUS)/goal/$(LAST) prepare.py $(RAW)
	mkdir -p $(MMAP_FILES)/focus-goal
	$(PYTHON) prepare.py $(FOCUS)/goal/ $(DK_DATA) $(MMAP_FILES)/focus-goal
	ln -sfn $(wildcard $(abspath $(MMAP_FILES)/full)/val.*.db) $(MMAP_FILES)/focus-goal/

$(NGRAW): $(FOCUS)/not-goal/$(LAST) prepare.py $(RAW)
	mkdir -p $(MMAP_FILES)/focus-not-goal
	$(PYTHON) prepare.py $(FOCUS)/not-goal/ $(DK_DATA) $(MMAP_FILES)/focus-not-goal
	ln -sfn $(wildcard $(abspath $(MMAP_FILES)/full)/val.*.db) $(MMAP_FILES)/focus-not-goal/

clean:
	rm -f $(RAW) $(SRAW) env
