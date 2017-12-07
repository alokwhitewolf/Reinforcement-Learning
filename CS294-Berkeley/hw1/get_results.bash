#!/bin/bash
set -eux
echo "Cloning Started"
for e in Hopper-v1 Ant-v1 HalfCheetah-v1 Humanoid-v1 Reacher-v1 Walker2d-v1
do
	echo cloning $e
	echo "##########################" >> cloning_results.txt
	echo $e >> cloning_results.txt
	echo "##########################" >> cloning_results.txt
	for n in 10 15 20 25 
		do
			python3 cloning.py experts/$e.pkl $e rollout_data/${e}_${n}_data.pkl --num_rollouts=$n
		done
	
done
echo "Cloning Done"
echo "DAgger Started"
for e in Hopper-v1 Ant-v1 HalfCheetah-v1 Humanoid-v1 Reacher-v1 Walker2d-v1
	do
			echo cloning $e
			echo "##########################" >> DAgger_results.txt
			echo $e >> DAgger_results.txt
			echo "##########################" >> DAgger_results.txt
			for n in 10 15 20 25
				do
					python3 Dagger.py experts/$e.pkl $e rollout_data/${e}_${n}_data.pkl --num_rollouts=$n
				done
		
	done

echo "DAgger Done"


