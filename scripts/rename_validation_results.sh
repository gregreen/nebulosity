cd ~/projects/nebulosity/validation_res

N=`ls -1 *.jpeg | wc -l`
#for (( k=1; k<$N; k++ )); do f=`ls _${k}_*.jpeg`; read p; p=`awk '{ print $3 }' $p`; p=${p#0.}; echo $f $p; mv $f ${p}.jpg; done < predictions.txt
for (( k=1; k<$N; k++ )); do
    f=`ls _${k}_*.jpeg`
    read p
    p=`echo $p | awk '{ print $3 }'`
    p=${p#0.}
    echo "origin=\"${f}\" dest=\"${p}.jpg\""
    mv $f ${p}.jpg
done < predictions.txt

montage *.jpg -geometry '64x64>+2+2' validation.jpg
montage 0*.jpg 1*.jpg 2*.jpg 3*.jpg 4*.jpg -geometry '128x128>+2+2' validation_nebulosity.jpg
montage 5*.jpg 6*.jpg 7*.jpg 8*.jpg 9*.jpg -geometry '128x128>+2+2' validation_normal.jpg
