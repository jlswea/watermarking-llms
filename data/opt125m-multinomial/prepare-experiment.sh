#!/bin/zsh

## @1 : Base path of experiment dir
## @2 : Decoding strategy

gamm_val_list=(25 50 75 90)
delta_val_list=(0 1 2 5 10)
server_base_path="/home/USER/watermarking-llms/"
sample_size="500"

for gamma_val in ${gamm_val_list}; do
    for delta_val in ${delta_val_list}; do
        experiment_dir="${2}-gamma${gamma_val}-delta${delta_val}-opt125m-opt350m"
        experiment_dir_path="$1/${experiment_dir}/"
        echo "${experiment_dir_path}"
        mkdir ${experiment_dir_path}
        realnews_path="${experiment_dir_path}realnews-sample.jsonl"
        echo ${realnews_path}
        shuf -n ${sample_size} /Users/USER/realnews/realnews.jsonl > ${realnews_path}

        gamma_dec=$((gamma_val / 100.0))
        echo ${gamma_dec}
        argument_str="-i ${server_base_path}conda-activate.sh cuWLLM ${server_base_path}wllm-pipeline.py -s ${sample_size} -v 2 -d cuda --green-list-bias ${delta_val} --green-list-size ${gamma_dec:0:4} --decoding-strategy $2 --in-data ${server_base_path}data/${experiment_dir}/realnews-sample.jsonl --out-data ${server_base_path}data/${experiment_dir}/archive.jsonl"
        echo ${argument_str}
        cat > "${experiment_dir_path}pipeline.sub" << END
            executable              = /bin/bash
            arguments               = ${argument_str}
            log                     = log.txt
            output                  = stdout.txt
            error                   = stderr.txt
            request_gpus            = 1
            queue

END
    done
done
