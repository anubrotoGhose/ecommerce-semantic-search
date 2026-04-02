import json

input_file = "../datasets/meta_Amazon_Fashion.jsonl"        # your original JSONL file
output_file = "../datasets/meta_Amazon_Fashion_10000.jsonl" # new file

max_rows = 10_000

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin):
        if i >= max_rows:
            break
        json.loads(line)          # validates JSON
        fout.write(line)

print("✅ Saved first 10,000 rows to", output_file)
