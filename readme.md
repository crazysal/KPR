

### Train
- data_list_basic_00 :
python main.py --loss mse --model basic --in_chanl 1 --in_sz 256 --targ_sz 256 --t_batch_size 200 --v_batch_size 1  --mode train  --epoch 20  --rd /home/sahmed9/reps/samsung/image-colorization_d6a566  --ird /home/sahmed9/reps/samsung/train_landscape_images/landscape_images/ --cache_dir ./cache/ --cache_file data_list_basic_00  --resume None  --test ./test_pose/  --op_dir ./cache/output/  --val_ep 4  --lr 1e-3  --cuda True  --saver 1  --scheduler 15
  

- EXP (9)
  CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py --conf_train 9 --conf_val 9v 

  CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py --conf_test 9t
  

- EXP (9a)
  CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py --conf_train 9a --conf_val 9av 

  CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py --conf_test 9at