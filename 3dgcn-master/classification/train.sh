python3 main.py \
-mode train \
-support 1 \
-neighbor 20 \
-cuda 0 \
-epoch 15 \
-bs 8 \
-dataset ModelNet40_1024_points \
-record record.log \
-save model.pkl \
#-normal

# python main.py -normal