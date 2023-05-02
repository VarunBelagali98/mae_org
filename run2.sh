CUDA_VISIBLE_DEVICES=2 python main_finetune.py \
    --accum_iter 1 \
    --batch_size 512 \
    --model vit_tiny_ours \
    --finetune /nfs/bigdisk/varun/save/amt/ent/checkpoint-399.pth \
    --dist_url tcp://localhost:10006 \
    --data_path /data/add_disk0/varun/imagenet100/ \
    --output_dir /nfs/bigdisk/varun/save/amt/ent_ft/ \
    --log_dir /nfs/bigdisk/varun/save/amt/ent_ft/ \
    --nb_classes 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    2>&1 | tee /nfs/bigdisk/varun/save/amt/adamae/ent_ft_dataaug.txt
