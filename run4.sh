CUDA_VISIBLE_DEVICES=2 python main_pretrain_log.py  --batch_size 512 \
    --world_size 1 \
    --accum_iter 1 \
    --model mae_vit_ours \
    --norm_pix_loss \
    --mask_ratio 0.50 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /data/add_disk0/varun/imagenet100/ \
    --dist_url tcp://localhost:10008 \
    --output_dir /nfs/bigdisk/varun/save/amt/ent_mr_0.5 \
    --log_dir /nfs/bigdisk/varun/save/amt/ent_mr_0.5 \
    --run_name ent_mr_0.5 \
    --num_workers 4 \
    --log_to_wandb 1
    #--resume /nfs/bigdisk/varun/save/amt/ent/checkpoint-40.pth
    
    #2>&1 | tee /nfs/bigdisk/varun/save/amt/mae_inr/mae_inr.txt

