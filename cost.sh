for m in swin_t swin_s swin_b swin_l cait_xxs24 pit vit 
do
	python measure_cost.py --model ${m} --is_Coord --is_SPT --is_LSA --type flops --dataset T-IMNET
done
