#for s in 0 1 2
# python main.py --disable_aug --disable_reg --is_SCL --gpu 0 --seed ${s}
# python main.py --model swin_t --lr 0.001 --disable_aug --disable_reg --is_SCL --gpu 1 --seed ${s}
# python main.py --disable_aug --disable_reg --gpu 2 --seed ${s}
#python main.py --model swin_t --lr 0.001 --disable_aug --disable_reg --gpu 0 --seed ${s}
python main.py --disable_reg --disable_aug --tag only_SPT_n --is_SPT --gpu 0 --model vit
python main.py --disable_reg --disable_aug --tag only_SPT_n --is_SPT --gpu 0 --model swin_t --lr 0.001
python main.py --disable_reg --disable_aug --tag only_LSA_n --is_LSA --gpu 0 --model swin_t --lr 0.001
python main.py --disable_reg --disable_aug --tag only_Coord_n --is_Coord --gpu 0 --model swin_t --lr 0.001
# python main.py --disable_reg --disable_aug --gpu 1 --model res110 --lr 0.001