# running on 0
# qwen + qt + 0 + 42
python /app/src/LLM_ss_inference.py --name "ss_qwen_qt_0_42" --seed 42 \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" --num_examples 0 \
    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# qwen + reddit + 0 + 42
python /app/src/LLM_ss_inference.py --name "ss_qwen_red_0_42" --seed 42 \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" --num_examples 0 \
    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# qwen + rip + 0 + 42
python /app/src/LLM_ss_inference.py --name "ss_qwen_rip_0_42" --seed 42 \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" --num_examples 0 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# qwen + qt + 3 + 42
python /app/src/LLM_ss_inference.py --name "ss_qwen_qt_3_42" --seed 42 \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" --num_examples 3 \
    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# qwen + reddit + 3 + 42
python /app/src/LLM_ss_inference.py --name "ss_qwen_red_3_42" --seed 42 \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" --num_examples 3 \
    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# qwen + rip + 3 + 42
python /app/src/LLM_ss_inference.py --name "ss_qwen_rip_3_42" --seed 42 \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" --num_examples 3 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# qwen + qt + 5 + 42
python /app/src/LLM_ss_inference.py --name "ss_qwen_qt_5_42" --seed 42 \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" --num_examples 5 \
    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# qwen + reddit + 5 + 42
python /app/src/LLM_ss_inference.py --name "ss_qwen_red_5_42" --seed 42 \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" --num_examples 5 \
    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# qwen + rip + 5 + 42
python /app/src/LLM_ss_inference.py --name "ss_qwen_rip_5_42" --seed 42 \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" --num_examples 5 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"
