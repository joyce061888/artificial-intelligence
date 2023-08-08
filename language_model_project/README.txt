How to run the probe task:

1. Use tv_prompts.tsv file as the main function's argument in the command line using probe_task.py
2. This will generate the results file called tv_prompts_results.tsv. Use this outputted results tsv file
   to generate the results of the GPT-J token query model's probability distributions by using this file
   as the input in the command line for gptj_scoring.py.