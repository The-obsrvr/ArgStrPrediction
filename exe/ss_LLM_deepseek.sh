# running on 0
## Deepseek + tester
#python /app/src/LLM_ss_inference.py --name "ss_deepseek_test" --seed 42 \
#    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 3 \
#    --data "/app/Data/tester.csv" --context_examples "/app/Data/qt30_context_examples.json" \
#    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# Deepseek + qt + 0 + 42
#python /app/src/LLM_ss_inference.py --name "ss_deepseek_qt_0_42" --seed 42 \
#    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 0 \
#    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
#    --do_inference --config "/app/src/config/ss_llm_config.yaml"

## Deepseek + reddit + 0 + 42
#python /app/src/LLM_ss_inference.py --name "ss_deepseek_red_0_42" --seed 42 \
#    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 0 \
#    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
#    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# Deepseek + rip + 0 + 42
python /app/src/LLM_ss_inference.py --name "ss_deepseek_rip_0_42" --seed 42 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 0 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"
#

# gemma + rip + 0 + 42
python /app/src/LLM_ss_inference.py --name "ss_gemma_rip_0_42" --seed 42 \
    --model_name_or_path "google/gemma-3-27b-it" --num_examples 0 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# gpt + rip + 0 + 42
python /app/src/LLM_ss_inference.py --name "ss_gpt_rip_0_42" --seed 42 \
    --model_name_or_path "unsloth/gpt-oss-20b-unsloth-bnb-4bit" --num_examples 0 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# Deepseek + qt + 3 + 42
#python /app/src/LLM_ss_inference.py --name "ss_deepseek_qt_3_42" --seed 42 \
#    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 3 \
#    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
#    --do_inference --config "/app/src/config/ss_llm_config.yaml"
#
## Deepseek + reddit + 3 + 42
#python /app/src/LLM_ss_inference.py --name "ss_deepseek_red_3_42" --seed 42 \
#    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 3 \
#    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
#    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# Deepseek + rip + 3 + 42
python /app/src/LLM_ss_inference.py --name "ss_deepseek_rip_3_42" --seed 42 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 3 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"
#

# gemma + rip + 3 + 42
python /app/src/LLM_ss_inference.py --name "ss_gemma_rip_3_42" --seed 42 \
    --model_name_or_path "google/gemma-3-27b-it" --num_examples 3 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# gpt + rip + 3 + 42
python /app/src/LLM_ss_inference.py --name "ss_gpt_rip_3_42" --seed 42 \
    --model_name_or_path "unsloth/gpt-oss-20b-unsloth-bnb-4bit" --num_examples 3 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# Deepseek + qt + 5 + 42
#python /app/src/LLM_ss_inference.py --name "ss_deepseek_qt_5_42" --seed 42 \
#    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 5 \
#    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
#    --do_inference --config "/app/src/config/ss_llm_config.yaml"
#
## Deepseek + reddit + 5 + 42
#python /app/src/LLM_ss_inference.py --name "ss_deepseek_red_5_42" --seed 42 \
#    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 5 \
#    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
#    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# Deepseek + rip + 5 + 42
python /app/src/LLM_ss_inference.py --name "ss_deepseek_rip_5_42" --seed 42 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --num_examples 5 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"
#

# gemma + rip + 5 + 42
python /app/src/LLM_ss_inference.py --name "ss_gemma_rip_5_42" --seed 42 \
    --model_name_or_path "google/gemma-3-27b-it" --num_examples 5 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# gpt + rip + 5 + 42
python /app/src/LLM_ss_inference.py --name "ss_gpt_rip_5_42" --seed 42 \
    --model_name_or_path "unsloth/gpt-oss-20b-unsloth-bnb-4bit" --num_examples 5 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"



## gemma + us2016 + 0+ 42
## gemma + qt30 + 0 + 42
## gemma + rip + 0 + 42
#
## gemma + us2016 + 3 +  42
## gemma + qt30 + 3 + 42
## gemma + rip + 3 + 42
#
## gemma + us2016 + 5 + 42
## gemma + qt30 + 5 + 42
## gemma + rip + 5 + 42
#
## gpt + us2016 + 0 42
## gpt + qt30 + 0 + 42
## gpt + rip + 0 + 42
#
## gpt + us2016 + 3 + 42
## gpt + qt30 + 3 +  42
## gpt + rip + 3 + 42
#
## gpt + us2016 + 5 + 42
## gpt + qt30 + 5 + 42
## gpt + rip + 5 + 42
