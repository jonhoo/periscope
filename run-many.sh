#!/bin/sh
run() {
	ex="$1"
	echo "Running experiment $ex" > /dev/stderr
	git --no-pager diff --color
	make solve
	mv network-large.mdl "network-${ex}.mdl"
	mv plot-large.png "plot-${ex}.png"
}

shutdown="no"
if [ $# -eq 1 ]; then
	if [ "$1" = "-s" ]; then
		shutdown="yes"
	fi
fi

ran=0
for f in *.run.patch; do 
	[ -e "$f" ] || exit 1

	git reset --hard
	if [ -s "$f" ]; then
		git apply "$f"
	fi
	run "${f%.run.patch}"
	ran=1
done

if [ $ran -eq 1 ]; then
	git reset --hard
	if [ $shutdown = "yes" ]; then
		sudo poweroff
	fi
fi
