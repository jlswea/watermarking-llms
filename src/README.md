# Code

| module | description |
| --- | --- |
| wllm.py | the CLI, entrypoint for running experiments |
| experiments.py | definition for individual experiments |
| pipeline.py | text generation pipeline |
| processor.py | implementation of the Huggingface LogitsProcessor class for embedding the watermark|
| archive.py | persistence of results, retrieval of metrics |