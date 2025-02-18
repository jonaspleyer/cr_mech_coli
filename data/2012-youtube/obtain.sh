YT_LINK=https://www.youtube.com/watch?v=4grQSLmWXQk
OUT_FILE=4grQSLmWXQk.mp4

yt-dlp $YT_LINK -o $OUT_FILE

mkdir frames

ffmpeg -i $OUT_FILE frames/%06d.png
