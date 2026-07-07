#!/bin/zsh

## @1 : Base path of experiment dir

decoding="beam"
beam_val_list=(1 4 8)
delta_val_list=(0 0.1 0.5 1 2 5 10)
gamma_val=0.5
server_base_path="/home/USER/watermarking-llms/"
sample_size="500"
model="facebook/opt-1.3b"
oracle_model="facebook/opt-2.7b"

for beam_val in ${beam_val_list}; do
    for delta_val in ${delta_val_list}; do
        experiment_dir="${decoding}${beam_val}-gamma${gamma_val}-delta${delta_val}-opt1.3b-opt2.7b"
        experiment_dir_path="$1/${experiment_dir}/"
        echo "${experiment_dir_path}"
        mkdir ${experiment_dir_path}
        realnews_path="${experiment_dir_path}realnews-sample.jsonl"
        echo ${realnews_path}
        shuf -n ${sample_size} /Users/USER/realnews/realnews.jsonl > ${realnews_path}

        cat > "${experiment_dir_path}config.ini" <<-EOF
		[Pipeline]
		sample_size=${sample_size}
		verbosity=0
		device=cuda 
		--green-list-bias=${delta_val}
		--green-list-size=${gamma_val}
		--decoding-strategy=$2 
		--num-beams=${beam_val}
		--in-data=./realnews-sample.jsonl
		--out-data=./archive.jsonl
		model=${model}
		oracle_model=${oracle_model}
		EOF
    done
done

cat > "$1/pipeline.sub" << EOF
executable              = /bin/bash
arguments               = -i ${server_base_path}conda-activate.sh cuWLLM ${server_base_path}src/wllm.py compare --from-config 
log                     = log.txt
output                  = stdout.txt
error                   = stderr.txt
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
initialdir              = \$(directory)
transfer_input_files    = config.ini, realnews-sample.jsonl
request_gpus            = 1
queue directory matching ${decoding}*
EOF

