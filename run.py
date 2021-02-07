import os
import tensorflow as tf
import imageio

os.system("python infer.py --model_dir=datasets/font27 \
                --batch_size=1 \
                --source_obj=save_dir/train.obj  \
                --embedding_ids=1,27 \
                --save_dir=output_dir  \
                --interpolate=0  \
                --steps=1")