python train_eval.py -m vit_base_patch16_224.augreg_in21k --head_dim_type task_classes --logit_type head_out \
     --training_string prompt head --prompt_len 4 --temperature 30 --null_thres_mode adaptive --refine_head True --impt_enable True --impt_topk 50 --impt_batch_size 100 --impt_strategy 1 --impt_loss_new SupCon --impt_select_level dime --impt_more_relax 0.04 \
     -b 240 -t 10 -d imagenet_r --data_root "[DATA_ROOT]" $@
     # -t: number of tasks; -d: dataset_name; --data_root: data dir containing 'train' and 'val'
