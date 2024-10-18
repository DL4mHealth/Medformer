export CUDA_VISIBLE_DEVICES=0,1,2,3

# Subject-Dependent
# ADFTD Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFTD/ \
  --model_id ADFTD-Dep \
  --model Transformer \
  --data ADFTD-Dependent \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10


# Subject-Independent
# APAVA Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/APAVA/ \
  --model_id APAVA-Indep \
  --model Transformer \
  --data APAVA \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# TDBRAIN Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/TDBRAIN/ \
  --model_id TDBRAIN-Indep \
  --model Transformer \
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

# ADFTD Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFTD/ \
  --model_id ADFTD-Indep \
  --model Transformer \
  --data ADFTD \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# PTB Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB/ \
  --model_id PTB-Indep \
  --model Transformer \
  --data PTB \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# PTB-XL Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB-XL/ \
  --model_id PTB-XL-Indep \
  --model Transformer \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size 256 \
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
  --model Transformer \
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
  --model Transformer \
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