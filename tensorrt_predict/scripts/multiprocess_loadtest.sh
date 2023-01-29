# Created by liujiarun@didiglobal.com 2022.12.25
SCRIPTS_DIR=$(cd `dirname $0`; pwd)

dev_num_list="0"
thread_number_list="1"

for dev_num in $dev_num_list
do
	for number in $process_number_list
	do
		./build/test/loadtest /home/pilot/jiarun.liu/workspace/models/faster-rcnn/faster-rcnn-trt8.4.engine 375 500 50 /home/pilot/jiarun.liu/workspace/data/test_375_500.jpg $number false $dev_num
	done
done
