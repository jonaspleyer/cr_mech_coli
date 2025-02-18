YT_LINK=https://www.youtube.com/watch?v=gEwzDydciWc
OUT_FILE=gEwzDydciWc.mp4

yt-dlp $YT_LINK -o $OUT_FILE

mkdir frames

ffmpeg -i $OUT_FILE frames/%06d.png
