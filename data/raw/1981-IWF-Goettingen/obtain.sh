DOWNLOAD_LINK=https://tib.flowcenter.de/mfc/medialink/3/deae5697401cd4105db462f076cac07710b1c8fe0864bfc4b9ef03ab8f3588e957/VTS_01_1_131.mp4
OUT_FILE=1981-goettingen-10.3203_IWF_K-129.mp4

# Download movie
if [ -f $OUT_FILE ]; then
    echo Using existing movie file $OUT_FILE
else
    wget $DOWNLOAD_LINK
    mv VTS_01_1_131.mp4 $OUT_FILE
fi

# Extract individual frames
N_FILES=$(ls -1q frames/* | wc -l)
if [ $N_FILES -eq 2207 ]; then
    echo Using existing files ./frames
else
    mkdir frames
    ffmpeg -i $OUT_FILE frames/%06d.png
fi

# Create First part of movie
mkdir -p growth-1
for i in $(seq 389 997); do
    cp frames/$(printf "%06d" $i).png growth-1
done

# Create Second part of movie
mkdir -p growth-2
for i in $(seq 998 1295); do
    cp frames/$(printf "%06d" $i).png growth-2
done

# Create Last part of movie
mkdir -p growth-3
for i in $(seq 1296 2136); do
    cp frames/$(printf "%06d" $i).png growth-3
done
