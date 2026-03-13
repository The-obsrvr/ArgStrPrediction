## single 42 CV Average Loss: 0.1015
             #2026-01-19 09:37:20,968 - INFO - >> Best Validation Loss: 0.1015
             #>> Best model saved to experiments/lm_ss_42_20260119_092306/best_bilstm_er_model.pth
             #2026-01-19 09:37:21,063 - INFO - >> Best model saved to experiments/lm_ss_42_20260119_092306/best_bilstm_er_model.pth


#finetuning
python /app/src/LM_ss_finetuning.py --name "lm_ss" --seed 42 \
    --data "/app/Data/QT30_training.csv" --config "/app/src/config/ss_lm_config.yaml"

# inference QT 42
#python /app/src/LM_ss_inference.py  --seed 42 \
#    --run_dir "/app/experiments/lm_ss_123" \
#        --data_path "/app/Data/QT30_test.csv"
#
## inference red 42
#python /app/src/LM_ss_inference.py  --seed 42 \
#        --data_path "/app/Data/US2016reddit.csv" --run_dir "/app/experiments/lm_ss_123"

# inference rip 42
python /app/src/LM_ss_inference.py  --seed 42 \
        --data_path "/app/Data/RIP1.csv" --run_dir "/app/experiments/lm_ss_123"

