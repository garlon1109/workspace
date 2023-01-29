#!/bin/bash

set -e

test-routine() {
    if [[ -n $1 && $BIND_CPU == 1 ]]
    then
        TASKSET_CMD="taskset -c $1"
    fi

    # cd test/pytorch/classification
    # $TASKSET_CMD python train.py --cifar10 --epoch=1 --print-freq=100 --model=vgg16
    cd test/pytorch/classification
    $TASKSET_CMD python train2.py --epoch=2 --batch-size="$BATCH_SIZE" --lr=0.1
    # cd test/pytorch
    # $TASKSET_CMD python mnist.py --epoch=2
}

BATCH_SIZE=64

if ! OPTIONS=$(getopt -o '' -l reef,two,bindcpu,batch-size: -- "$@")
then
    exit 1
fi

eval set -- "$OPTIONS"
while true
do
    case $1 in
        --reef)
            echo "Option: Use REEF"
            USE_REEF=1
            shift
        ;;
        --two)
            echo "Option: Launch two tasks"
            TWO_TASK=1
            shift
        ;;
        --bindcpu)
            echo "Option: Bind CPU"
            BIND_CPU=1
            shift
        ;;
        --batch-size)
            echo "Option: Batch size $2"
            BATCH_SIZE=$2
            shift 2
        ;;
        --)
            shift
            break
        ;;
        *)
            echo "Invalid option: $1" >&2
            exit 1
        ;;
    esac
done

if [[ -n $USE_REEF ]]
then
    export LD_LIBRARY_PATH=$PWD/output/lib:$LD_LIBRARY_PATH
    export LD_PRELOAD=$PWD/output/lib/libcudareef.so
fi

CLI_TOML_PATH=$PWD/scheduler/cli/Cargo.toml

export NOBANNER=1

if [[ -z $TWO_TASK ]]
then
    test-routine 0
else
    test-routine 1 &
    test-routine 2 &

    if [[ -n $USE_REEF ]]
    then
        PID=$(pgrep -n python)

        if [[ -z $PID ]]
        then
            echo "Error"
            exit 1
        fi

        echo "Choose $PID as backgroud process"
        cargo run --manifest-path "$CLI_TOML_PATH" -- set -p "$PID" -v 10
    fi

    wait
    echo "Both process ended"
fi