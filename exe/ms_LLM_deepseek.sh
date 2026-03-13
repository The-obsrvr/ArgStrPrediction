#running on 3
# Deepseek + qt + 0 + 100
python /app/src/LLM_ms_inference.py --name "ms_deepseek_qt_0_100" --seed 100 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 0 \
    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ms_llm_config.yaml"

# Deepseek + reddit + 0 + 100
python /app/src/LLM_ms_inference.py --name "ms_deepseek_red_0_100" --seed 100 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 0 \
    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ms_llm_config.yaml"

# Deepseek + rip + 0 + 100
python /app/src/LLM_ms_inference.py --name "ms_deepseek_rip_0_100" --seed 100 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 0 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ms_llm_config.yaml"

# Deepseek + qt + 3 + 100
python /app/src/LLM_ms_inference.py --name "ms_deepseek_qt_3_100" --seed 100 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 3 \
    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ms_llm_config.yaml"

# Deepseek + reddit + 3 + 100
python /app/src/LLM_ms_inference.py --name "ms_deepseek_red_3_100" --seed 100 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 3 \
    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ms_llm_config.yaml"

# Deepseek + rip + 3 + 100
python /app/src/LLM_ms_inference.py --name "ms_deepseek_rip_3_100" --seed 100 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 3 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ms_llm_config.yaml"

# Deepseek + qt + 5 + 100
python /app/src/LLM_ms_inference.py --name "ms_deepseek_qt_5_100" --seed 100 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 5 \
    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ms_llm_config.yaml"

# Deepseek + reddit + 5 + 100
python /app/src/LLM_ms_inference.py --name "ms_deepseek_red_5_100" --seed 100 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 5 \
    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ms_llm_config.yaml"

# Deepseek + rip + 5 + 100
python /app/src/LLM_ms_inference.py --name "ms_deepseek_rip_5_100" --seed 100 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 5 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ms_llm_config.yaml"