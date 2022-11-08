#!/bin/bash

for (( i=$1; i<$2; i++ ))
do
   echo "Starting $i..."
	 sbatch model_building_order2.sh 1000 $i
done
