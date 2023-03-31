RND=$RANDOM
lockedfile=runpost${RND}.sh
cp runpost.sh $lockedfile
for icount in {0..20..1}; do
echo $icount
( source $lockedfile; wait ) & wait
#source runpost.sh& 
# Get the PID of the child process
pid=$!

# Wait for the child process to complete
wait $pid
#sleep 1m # wait for 1 min

done

sudo shutdown +5
~                                           
