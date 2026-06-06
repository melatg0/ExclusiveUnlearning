MODEL_NAME="Llama-3.2-1B-Instruct"
MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
MODEL_DATA="Llama-3.2-1B-Instruct"

# Root directories. Override by exporting DATA_DIR / OUT_ROOT before running,
# e.g. `DATA_DIR=/mnt/data OUT_ROOT=/mnt/out bash run_exclusive_unlearning_metamath.sh`
DATA_DIR="${DATA_DIR:-data}"
OUT_ROOT="${OUT_ROOT:-out}"
OUT_DIR="${OUT_ROOT}/math_sft/metamath/${MODEL_NAME}"


python src/exclusive_unlearning/run_exclusive_unlearning.py \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUT_DIR}/" \
  --per_device_train_batch_size 4 \
  --train_generate_batch_size 4 \
  --learning_rate 1e-5 \
  --num_train_steps 10000 \
  --logging_steps 50 \
  --report_to wandb \
  --wandb_project "math_sft_${MODEL_NAME}_entropymax" \
  --forget_sample_file "${DATA_DIR}/forget_samples/${MODEL_DATA}/sampled_texts.jsonl" \
  --run_name "${MODEL_NAME}-math_sft" \
  --eval_text_file "${DATA_DIR}/forget_samples/${MODEL_DATA}/sampled_texts_for_evaluate_loss.jsonl" \
  --forget_lambda 0.4 \
  --seq_len 256 \
  --retain_train_data_file "${DATA_DIR}/MetaMathQA/train_split.json" \
  --retain_eval_data_file "${DATA_DIR}/MetaMathQA/test.jsonl"