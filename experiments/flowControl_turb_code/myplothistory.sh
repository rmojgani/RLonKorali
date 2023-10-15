# viewout.sh : grep -wrin "Average Reward for agent 0" training_CASE1_*.out
bash viewout.sh | \grep -o '...........$'>myhistory.out

gnuplot <<- EOF
    set terminal dumb
    set xlabel "Generation"
    set ylabel "Reward"
    set title "Reward history"
    set output 'myscatterplot.plot'
    set logscale y
   # set margins 0,0,0,0
   # set autoscale xfix
   # set autoscale yfix
    unset xtics
   # unset ytics
   # unset key
    plot "myhistory.out" with points pointtype "."
EOF
vim myscatterplot.plot
