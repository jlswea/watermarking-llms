import pandas as pd


def get_long_prompt(path):
    max_len = 0
    max_line = ""
    json = pd.read_json(path_or_buf=path, lines=True)
    text = json.get("text")
    for line in text:
        if len(line) > max_len:
            max_line = line
            max_len = len(line)

    return max_line
