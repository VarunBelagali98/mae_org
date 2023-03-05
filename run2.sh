CUDA_VISIBLE_DEVICES=0,1 python main_pretrain_wp.py  --batch_size 64 \
    --world_size 1 \
    --accum_iter 32 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /home/xiangli/imagenet \
    --dist_url tcp://localhost:10006 \
    --policy_config ./configs/p1.yaml \
    --mode random \
    --output_dir ../save/random_mae \
    --log_dir ../save/random_mae \
    --resume ../save/random_mae/checkpoint-320.pth \
    2>&1 | tee ../save/logs/random_mae.txt
