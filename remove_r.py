lines = open('tune_and_train_llama2.sh')
for line in lines:
  if '\r' in line:
    print(line)


