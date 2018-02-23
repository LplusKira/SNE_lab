#!/usr/local/bin/gnuplot
set terminal dumb
set datafile sep ','
set title myTitle 
set xlabel xaxis

plot filename using 1:2
