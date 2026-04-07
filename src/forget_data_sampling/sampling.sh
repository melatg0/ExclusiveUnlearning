MODEL_NAME=Llama-3.2-1B-Instruct
MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct
python src/forget_data_sampling/sampling.py \
  --model_name_or_path ${MODEL_PATH} \
  --output_file data/forget_samples/${MODEL_NAME}/sampled_texts.jsonl \
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