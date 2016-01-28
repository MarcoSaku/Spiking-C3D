#!/usr/bin/env bash

youtubedir=/media/TB/Videos/youtube-objects-all
#ls ${youtubedir}/*.avi | xargs -I {} ffmpeg -i {} -vcodec copy -acodec copy -f null /dev/null 2>&1 | grep 'frame=' | awk '{print $2}'
inputfile=/home/chuck/projects/C3D/examples/c3d_feature_extraction/prototxt/youtube_objects_input_list_video.txt
outputfile=/home/chuck/projects/C3D/examples/c3d_feature_extraction/prototxt/youtube_objects_output_list_video_prefix.txt
outdir=/media/TB/Videos/youtube-objects-all/c3d_features
numframesc3d=16

minframenum=30

rm -f $inputfile $outputfile
rm -rf $outdir/*

FILES=${youtubedir}/*.avi
for f in $FILES
do
  basef=${f##*/}
  basefnoext=${basef%.*}
  echo "Processing $basef..."
  framenum=$(ffmpeg -i $f -vcodec copy -acodec copy -f null /dev/null 2>&1 | grep 'frame=' | awk '{print $2}')
  echo "#frame=${framenum}"

  if [ $framenum -le $minframenum ]; then
    echo "too few frames. Skipping this shot..."
    continue
  fi;

  curframe=0
  for i in $(seq -f "%05g" 0 $numframesc3d $((framenum - minframenum + 1)) )
  #for i in $(seq -f "%05g" 0 $numframesc3d $framenum)
  do
    echo "$f $(( 10#$i + 0 )) 0"
    echo "$outdir/$basefnoext/$i"

    echo "$f $(( 10#$i + 0 )) 0" >> $inputfile
    echo "$outdir/$basefnoext/$i" >> $outputfile

    mkdir -p $outdir/$basefnoext
  done

  #while [[ $curframe -le $framenum ]]
  #do
  #  echo "curframe=${curframe}"
  #  curframe=$(( $curframe + $numframesc3d ))
  #  echo $outdir/$f/
  #done

done
