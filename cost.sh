for m in coatnet_0
do
	python measure_cost.py --model ${m} --is_Coord --is_SPT --is_LSA --type flops --dataset T-IMNET
	python measure_cost.py --model ${m} --type flops --dataset T-IMNET

done
