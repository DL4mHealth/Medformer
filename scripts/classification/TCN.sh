export CUDA_VISIBLE_DEVICES=0,1,2,3


# TBRAIN Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/TBRAIN/ \
  --model_id TDBRAIN-Indep \
  --model TCN \
  --data TDBRAIN \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10


# Two human activities dataset

# UCI-HAR Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UCI-HAR/ \
  --model_id UCI-HAR \
  --model TCN \
  --data UCI-HAR \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10


# FLAAP Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FLAAP/ \
  --model_id FLAAP \
  --model TCN \
  --data FLAAP \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10