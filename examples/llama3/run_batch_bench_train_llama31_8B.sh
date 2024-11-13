#!/bin/bash


WORLD_SIZE=8
SEQ_LENGTHS_arr=(8192 16384)
TP_arr=(1 )
PP_arr=(1 2 )
DP_arr=(8 4 )
CP_arr=(1 )
MBS_arr=(1 2 4 6)
AC_arr=(none sel full)


export MODEL_SIZE=8
export SEQ_PARALLEL=1
export GBS=128
export DO=true
export FL=true
export TE_FP8=0
export TOTAL_ITERS=5
export MOCK_DATA=1
export RECOMPUTE_NUM_LAYERS=32


for i in $(seq 0 $[${#SEQ_LENGTHS_arr[@]}-1]); do
    export SEQ_LENGTHS=${SEQ_LENGTHS_arr[i]}
    
    for j in $(seq 0 $[${#CP_arr[@]}-1]); do
        export CP=${CP_arr[j]}

        for p in $(seq 0 $[${#DP_arr[@]}-1]); do
            export DP=${DP_arr[j]}

            for k in $(seq 0 $[${#PP_arr[@]}-1]); do
                export PP=${PP_arr[k]}

                for l in $(seq 0 $[${#TP_arr[@]}-1]); do
                    export TP=${TP_arr[l]}

                    if [[ $WORLD_SIZE -eq $((DP*TP*PP*CP)) ]]; then
                        for m in $(seq 0 $[${#AC_arr[@]}-1]); do
                            export AC=${AC_arr[m]}

                            for n in $(seq 0 $[${#MBS_arr[@]}-1]); do
                                export MBS=${MBS_arr[n]}

                                bash ./bench_train_llama3.sh

                                wait
                                sleep 5

                            done

                        done
                    else
                        echo "WORLD_SIZE != DP*TP*PP*CP"
                    fi  
                done

            done
        done 
    done

done