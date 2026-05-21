videos=("blackswan" "boat" "breakdance" "breakdance-flare" "bus" "car-roundabout" 
        "car-shadow" "car-turn" "dance-jump" "dance-twirl" "drift-chicane" "drift-straight" 
        "drift-turn" "elephant" "flamingo" "goat" "hike" "libby" "lucia" "mallard-water" "parkour" 
        "rhino" "rollerblade")

for video in "${videos[@]}"; do
    python run.py --input_video PATH_TO_DAVIS_VIDEO_DIRECTORY/${video}.mp4 --output_dir PATH_TO_DAVIS_DEPTH_DIRECTORY --encoder vitl --save_npz
done
