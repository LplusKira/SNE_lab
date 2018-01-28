#!/usr/local/bin/gnuplot
set terminal dumb
set title "qq" font ",14" textcolor rgbcolor "royalblue"
set xlabel "run"
set ylabel "microF1"

plot filename using 1:2
