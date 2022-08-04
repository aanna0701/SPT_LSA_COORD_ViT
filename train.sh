for s in 0 1 2
# python main.py --disable_aug --disable_reg --is_SCL --gpu 0 --seed ${s}
# python main.py --model swin_t --lr 0.001 --disable_aug --disable_reg --is_SCL --gpu 1 --seed ${s}
# python main.py --disable_aug --disable_reg --gpu 2 --seed ${s}
python main.py --model swin_t --lr 0.001 --disable_aug --disable_reg --gpu 0 --seed ${s}
