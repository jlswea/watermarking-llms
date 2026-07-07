# Data

| Dataset | Links |
| --- | --- | 
| RealNews | [Existing dataset and code](https://github.com/rowanz/grover/blob/master/realnews/README.md) | 
| C4 | [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/c4), [AllenAI processed version](https://huggingface.co/datasets/allenai/c4/blob/main/README.md) | 

## Preparation

Download and unpack the dataset. The unzipped file will be ~130GB. This is too big for usual amounts of RAM. Therefore either read in the file in chunks if the whole dataset is needed:
```python
with pd.read_json(filename, chunksize=chunksize) as reader:
    for chunk in reader:
        process(chunk)
```
Or create a subsample of the data:
```bash
shuf -n <number_of_lines_to_be_sampled> ./realnews.jsonl > realnes_sample.jsonl
```
This will shuffle the input file and output `n` lines to stdout, hence creating a randomly sampled subset. See this [Stackoverflow discussion](https://stackoverflow.com/questions/22258491/read-a-small-random-sample-from-a-big-csv-file-into-a-pandas-data-frame/48589768#48589768) for details.

### Automated experiment setup

1. Create a a dir <dirname> for the experiment.
2. Run `prepare-experiment.sh <dirname>`. Some changed to the script might be needed to tailor it for a given experiment.
3. Submit `pipelin.sub` to the HTCondor scheduler. This is a [late materialization submission](https://htcondor.readthedocs.io/en/latest/users-manual/submitting-a-job.html#submitting-lots-of-jobs) that will execute a job for all subdirectories regardless of `MAX_JOBS_PER_OWNER`.
4. Aggregate results of all jobs into a single `results.csv` by running `python src/archive.py get_results <dirname>`.
