#!/bin/sh
run() {
	ex="$1"
	make solve
	ln network-large.mdl "network-${ex}.mdl"
	ln plot.png "plot-${ex}.png"
	git diff
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

	echo "Using $ex" > /dev/stderr
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
		sudo shutdown -h now
	fi
fi
