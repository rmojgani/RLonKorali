RND=$RANDOM
lockedfile=run${RND}.sh
cp run.sh $lockedfile

for icount in {0..10..1}; do
echo $icount
( source $lockedfile; wait ) & wait

# Get the PID of the child process
pid=$!

# Wait for the child process to complete
wait $pid
#sleep 1m # wait for 1 min

done

sudo shutdown +5
