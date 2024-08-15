mkdir -p tuned/qwen1.5_7b
mv rocblas.yaml tuned/qwen1.5_7b/
mv full_tuned*.csv tuned/qwen1.5_7b/

export PYTORCH_TUNABLEOP_FILENAME=tuned/qwen1.5_7b/full_tuned%d.csv
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_ENABLED=1
bash train_qwen1.5_7b.sh
