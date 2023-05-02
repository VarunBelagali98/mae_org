CUDA_VISIBLE_DEVICES=2 python main_pretrain_log.py  --batch_size 16 \
    --world_size 1 \
    --accum_iter 1 \
    --model mae_inr_vit \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /data/add_disk0/varun/imagenet100/ \
    --dist_url tcp://localhost:10005 \
    --output_dir /nfs/bigdisk/varun/save/amt/mae_inr \
    --log_dir /nfs/bigdisk/varun/save/amt/mae_inr \
    --run_name adamae \
    --log_to_wandb 1 \
    2>&1 | tee /nfs/bigdisk/varun/save/amt/mae_inr/mae_inr.txt

    #--norm_pix_loss \