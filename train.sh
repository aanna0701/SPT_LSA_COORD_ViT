for s in 0 1 2
do
	python main.py --model vit --lr 0.003 --is_LSA --is_Coord --is_SPT --seed ${s} --gpu 0 --dataset CIFAR100 --data_path ../
#	python main.py --model swin --lr 0.001 --is_LSA --is_Coord --is_SPT --seed ${s} --gpu 0 --dataset CIFAR100 --data_path ../
	
done
