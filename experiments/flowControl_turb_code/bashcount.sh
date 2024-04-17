date +"%d-%m-%y"
for d in _result*/ ; do
    generation=$(ls $d/*.json | wc -l)
    post_count=$(ls $d/C*post/*.png | wc -l)
    echo "$d | generation:$generation , post count:$post_count"
done
