# Watermarking LLMs

Read the thesis online: https://jlswea.github.io/watermarking-llms/

Related research: [Kirchenbauer et al. 2023](https://arxiv.org/pdf/2301.10226.pdf)

## Python Environment
Import Conda environment: 
```bash
conda env create -n <name> --file env.yaml
```
Export environment after changes: 
```bash
conda env export --from-history > env.yaml
```

Also see [Conda - Managing Environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) and [Conda Cheatsheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf).

## Inference
Text generation with [OPT](https://huggingface.co/facebook/opt-125m) and the [HuggingFace Transformer](https://huggingface.co/docs/transformers/main/en/model_doc/opt#transformers.OPTForCausalLM) library with Pytorch backend:
```bash
python model.py
```
The following CLI flags are available:
```bash
  -h, --help            show this help message and exit
  -v {0,1,2}, --verbosity {0,1,2}
                        Verbosity level of stdout output. Ranges from 0, i.e., low verbosity to 2.
  -d {cpu,cuda}, --device {cpu,cuda}
                        Device used for expensive computations, mainly matrix multiplications.
  -s SAMPLE_SIZE, --sample-size SAMPLE_SIZE
                        Number of samples to be drawn from the data source.
  -m {facebook/opt-125m,facebook/opt-350m,facebook/opt-1.3b,facebook/opt-2.7b}, --model {facebook/opt-125m,facebook/opt-350m,facebook/opt-1.3b,facebook/opt-2.7b}
                        Model weights used to generate watermarked text.
  --oracle-model {facebook/opt-350m,facebook/opt-1.3b,facebook/opt-2.7b}
                        Oracle model weights used to compute perplexity of the watermarked text.
  --in-data IN_DATA     Path to the dataset used for generating prompts and baseline completions. Format needs to be jsonl.
  --out-data-dir OUT_DATA_DIR
                        Path to the directory for archiving generated output and statistics.
  --green-list-size {0.0-1.0}
                        Size of the green list as a ratio of the tokenizers vocabulary.
  --green-list-bias GREEN_LIST_BIAS
                        Delta value added to all green list token before applying softmax.
  --decoding-strategy {multinomial,greedy,beam}
                        Strategy for sampling token from the vocabulary given the models logit vector
  --completion-size COMPLETION_SIZE
                        Size of the input text that will be cut off from the end, to be used as a baseline completion.
  --num-beams NUM_BEAMS
                        Number of beams; Relevant only if decoding_strategy = beam.
```
## Remote execution
Experiments are executed on the [pascal](https://docs.l3s.uni-hannover.de/#/computing/clusters/pascal/index) gpu server.
Jobs are defined as a `*.sub` file using this template:
```text
executable              = /bin/bash
arguments               = -i conda-activate.sh cuWLLM wllm-pipeline.py -s 5 -v 0 --decoding-strategy greedy --in-data /home/USER/watermarking-llms/data/realnews_sample_1k.jsonl --out-data-dir /home/USER/watermarking-llms/data/archive/
log                     = wllm.log
transfer_input_files    = conda-activate.sh, wllm-pipeline.py 
output                  = stdout.txt
error                   = stderr.txt
should_transfer_files   = IF_NEEDED
request_gpus            = 1
queue
``` 
The job is then submitted to [HTCondor](https://docs.l3s.uni-hannover.de/#/guides/htcondor/index) for scheduling. This can only be done on one of the dedicated submit nodes:
```bash
condor_submit <template>.sub // add job to the queue
condor_q                     // list your jobs
condor_tail -f <jobId>       // view stdout of one job
``` 


## TODO
- Displaying only generated text without parts of the prompt does not always work exactly, as in: 
  ```text
  Prompt ends with: ...  time to investigate the number and found the actual number to be 15. It is
  Baseline completion starts with:         indicative of how gun control groups are willing to stretch the truth to a ...
  ------------------------------------------------------------------------------------------------------------------------
  Generated completation starts with:      important to remember that the media is not the only source of information ...
  Tagged completion starts with:          o be 15. It is important{g} to{r} remember{g} that{g} the{g} media{g} is{g} ...
  ```
  The decoded prompt seems to be longer than the prompts token list. Hence, some token seem to get added in the tokenizers decoding step. This is probably related to lots of unicode replacement characters `\ufffd` in the output.