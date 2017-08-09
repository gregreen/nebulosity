rm normal.txt
rm nebulosity.txt
rm nebulosity_light.txt
rm nebulosity_heavy.txt
rm bg_error.txt

for d in `ls -d a?/`; do
  echo Looking in $d ...
  for sd in `ls -d ${d}*/`; do
    sd=${sd#$d}
    fo=${sd%\/}.txt
    for f in `ls -1 ${d}${sd}*.jpg`; do
        f=${f#${d}${sd}}
        f=${f%.jpg}.png
        echo $f >> $fo
    done
  done
done

# Add "nebulosity_heavy" into "nebulosity"
echo "Merging nebulosity_heavy into nebulosity..."
cat nebulosity_heavy.txt >> nebulosity.txt

# Determine number of images in each category
echo ""
echo "normal           : `wc -l normal.txt | awk '{print $1}'`"
echo "nebulous         : `wc -l nebulosity.txt | awk '{print $1}'`"
echo "nebulous (light) : `wc -l nebulosity_light.txt | awk '{print $1}'`"
echo "nebulous (heavy) : `wc -l nebulosity_heavy.txt | awk '{print $1}'`"
echo "bg_error         : `wc -l bg_error.txt | awk '{print $1}'`"
echo ""

