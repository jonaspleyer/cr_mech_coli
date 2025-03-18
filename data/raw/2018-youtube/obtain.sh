YT_LINK=https://www.youtube.com/watch?v=hUt55R8qU9g
OUT_FILE=hUt55R8qU9g.mp4.mkv

yt-dlp $YT_LINK -o $OUT_FILE

mkdir frames

ffmpeg -i $OUT_FILE frames/%06d.png
