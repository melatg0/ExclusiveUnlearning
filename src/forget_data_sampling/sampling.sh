# Model. Override by exporting MODEL_PATH / MODEL_NAME, e.g.
# `MODEL_PATH=google/gemma-3-12b-it MODEL_NAME=gemma-3-12b-it bash sampling.sh`.
# Use gemma-2-2b-it for fast local debugging.
MODEL_NAME="${MODEL_NAME:-gemma-2-9b-it}"
MODEL_PATH="${MODEL_PATH:-google/gemma-2-9b-it}"

# Root directory for generated forget data. Override with DATA_DIR if needed.
DATA_DIR="${DATA_DIR:-data}"

python src/forget_data_sampling/sampling.py \
  --model_name_or_path ${MODEL_PATH} \
  --output_file "${DATA_DIR}/forget_samples/${MODEL_NAME}/sampled_texts.jsonl" \
  --user_prompt_mode generate \
  --num_train_steps 400 \
  --per_device_train_batch_size 100 \
  --user_length_choices "64" \
  --user_system_prompt "" \
  --assistant_system_prompt "" \
  --use_chat_template True \
  --assistant_len 128 \
  --temperature 2.0 \
  --top_k 100 \
  --seed 42 \
  --device cuda