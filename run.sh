for seed in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$3 python3.6 main.py --agent $1 --game $2 --enable-cudnn --seed $seed --id $1
done