#running on 0 4422,  Unsloth 2026.1.3: Fast gemma12_Oss patching. Transformers: 4.57.3.
              #   \\   /|    NVIDIA A40. Num GPUs = 1. Max memory: 44.549 GB. Platform: Linux.
              #O^O/ \_/ \    Torch: 2.9.1+cu128. CUDA: 8.6. CUDA Toolkit: 12.8. Triton: 3.5.1

## gemma12 + qt + 0 + 42
#python /app/src/LLM_ss_inference.py --name "ss_gemma12_qt_0_42" --seed 42 \
#    --model_name_or_path "google/gemma-3-12b-it" --num_examples 0 \
#    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
#    --do_inference --config "/app/src/config/ss_llm_config.yaml"
#
## gemma12 + reddit + 0 + 42
#python /app/src/LLM_ss_inference.py --name "ss_gemma12_red_0_42" --seed 42 \
#    --model_name_or_path "google/gemma-3-12b-it" --num_examples 0 \
#    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
#    --do_inference --config "/app/src/config/ss_llm_config.yaml"
#
## gemma12 + rip + 0 + 42
#python /app/src/LLM_ss_inference.py --name "ss_gemma12_rip_0_42" --seed 42 \
#    --model_name_or_path "google/gemma-3-12b-it" --num_examples 0 \
#    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
#    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# gemma12 + qt + 3 + 42
python /app/src/LLM_ss_inference.py --name "ss_gemma12_qt_3_42" --seed 42 \
    --model_name_or_path "google/gemma-3-12b-it" --num_examples 3 \
    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# gemma12 + reddit + 3 + 42
python /app/src/LLM_ss_inference.py --name "ss_gemma12_red_3_42" --seed 42 \
    --model_name_or_path "google/gemma-3-12b-it" --num_examples 3 \
    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# gemma12 + rip + 3 + 42
python /app/src/LLM_ss_inference.py --name "ss_gemma12_rip_3_42" --seed 42 \
    --model_name_or_path "google/gemma-3-12b-it" --num_examples 3 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# gemma12 + qt + 5 + 42
python /app/src/LLM_ss_inference.py --name "ss_gemma12_qt_5_42" --seed 42 \
    --model_name_or_path "google/gemma-3-12b-it" --num_examples 5 \
    --data "/app/Data/QT30_test.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# gemma12 + reddit + 5 + 42
python /app/src/LLM_ss_inference.py --name "ss_gemma12_red_5_42" --seed 42 \
    --model_name_or_path "google/gemma-3-12b-it" --num_examples 5 \
    --data "/app/Data/US2016reddit.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"

# gemma12 + rip + 5 + 42
python /app/src/LLM_ss_inference.py --name "ss_gemma12_rip_5_42" --seed 42 \
    --model_name_or_path "google/gemma-3-12b-it" --num_examples 5 \
    --data "/app/Data/RIP1.csv" --context_examples "/app/Data/qt30_context_examples.json" \
    --do_inference --config "/app/src/config/ss_llm_config.yaml"