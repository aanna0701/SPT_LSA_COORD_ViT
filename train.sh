for s in 0 1 2
do
#	python main.py --model vit_s --batch_size 512 --lr 0.001 --epoch 75 --is_MAE --seed ${s} --gpu 0 --dataset CIFAR100 --data_path ../dataset
	python main.py --model cct_7 --lr 0.001 --seed ${s} --gpu 0 --dataset T-IMNET --data_path ../dataset
#	python main.py --model coatnet3_0 --lr 0.001 --seed ${s} --gpu 0 --dataset CIFAR100 --is_LSA --is_SPT --is_Coord --data_path ../dataset
	# python main.py --model egnet_400mf --lr 0.001 --seed ${s} --gpu 0 --dataset CIFAR100 --data_path ../datase	
done
