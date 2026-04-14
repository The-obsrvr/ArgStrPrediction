### running on 0
## lm_ms_42_20260120_121033 : current model directory.

# Finetuning
#python /app/src/LM_ms_finetuning.py --name "lm_ms" --seed 42 \
#    --data "/app/Data/QT30_training.csv" --config "/app/src/config/ms_lm_config.yaml"

## inference QT 42
#python /app/src/LM_ms_inference.py --run_name "lm_ms_42_20260120_121033" --seed 42 \
#    --model_name_or_path "/app/experiments/lm_ms_42_20260120_121033"  \
#        --data "/app/Data/QT30_test.csv" --config "/app/src/config/ms_lm_config.yaml"

# inference red 42
python /app/src/LM_ms_inference.py --name "lm_ms_42_20260120_121033" --seed 42 \
    --model_name_or_path "/app/experiments/lm_ms_42_20260120_121033"  \
        --data "/app/Data/US2016reddit.csv" --config "/app/src/config/ms_lm_config.yaml"


# inference rip 42
python /app/src/LM_ms_inference.py --name "lm_ms_42_20260120_121033" --seed 42 \
    --model_name_or_path "/app/experiments/lm_ms_42_20260120_121033"  \
        --data "/app/Data/RIP1.csv" --config "/app/src/config/ms_lm_config.yaml"
