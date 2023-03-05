python main_finetune.py \
    --batch_size 32 \
    --world_size 1 \
    --accum_iter 16 \
    --model vit_base_patch16 \
    --finetune ../save/random_mae/checkpoint-399.pth \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --output_dir ../save/debug \
    --log_dir ../save/debug \
    --data_path /home/xiangli/imagenet \
    --dist_url tcp://localhost:10006 \
    2>&1 | tee ../save/logs/debug.txt
