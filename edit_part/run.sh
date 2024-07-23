# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 train.py --config configs/skull/skull_geo.json
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 train.py --config configs/skull/skull_appear.json


# -----------------------------show config during drawing----------------------------------
# python show.py --config configs/deer/show_deer.json
# python show.py --config configs/cake/show_cake.json
# python show.py --config configs/cat/show_cat.json
# python show.py --config configs/chair/show_chair.json
# python show.py --config configs/wood/show_fly.json
# python show.py --config configs/pegasus/show_pegasus.json
# python show.py --config configs/flash/show_flash.json
# python show.py --config configs/headset/show_headset.json
# python show.py --config configs/labrador/show_labrador.json
# python show.py --config configs/naruto/show_naruto.json
# python show.py --config configs/rhino/show_rhino.json
# python show.py --config configs/rose/show_rose.json
# python show.py --config configs/skull/show_skull.json
# python show.py --config configs/stool/show_stool.json
# python show.py --config configs/turtle/show_turtle.json

