CUDA_VISIBLE_DEVICES=1,2 python main_pretrain_wp.py  --batch_size 1 \
    --world_size 1 \
    --accum_iter 4 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /data/add_disk0/varun/imagenet \
    --dist_url tcp://localhost:10005 \
    --policy_config ./configs/p1.yaml


    #--rdzv_endpoint=localhost:10009
    # torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 