import os
import tensorflow as tf
import imageio
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser(description='Generate font for single character')
parser.add_argument('--font_ids', dest='font_ids', help='directory that saves the model checkpoints', )
args = parser.parse_args()
font_ids = args.font_ids

print("Convert character to image...")

os.system("python font2img.py \
                --src_font=SIMSUN.ttf \
                --dst_font=SIMSUN.ttf \
                --charset=CN --sample_count=2 \
                --sample_dir=hanzi_dir \
                --label=0 \
                --filter=1 \
                --shuffle=1")

print("Convert to comply with generator input...")

os.system("python package.py \
                --dir=hanzi_dir \
                --save_dir=save_dir \
                --split_ratio=0")


print("Infer new font...")

os.system("python infer.py --model_dir=datasets/font27 \
                --batch_size=1 \
                --source_obj=save_dir/train.obj  \
                --embedding_ids="+str(font_ids)+" \
                --save_dir=output_dir  \
                --interpolate=0  \
                --steps=1 \
                --uroboros=1")
            