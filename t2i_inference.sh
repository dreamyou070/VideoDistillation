#srun -p big_suma_rtx3090 -q big_qos --gres=gpu:1 --pty bash -i --m 3 --is_teacher

python t2i_inference.py \
  --skip_layers " ['up_0_0','up_1_0','up_2_0','up_3_0','up_0_2','up_1_2','up_2_2','up_3_2','mid',]"