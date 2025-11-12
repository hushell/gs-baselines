exp_name='baseline'

voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1

ulimit -n 4096

./train.sh -d mipnerf360/bicycle -l ${exp_name}  --gpu 0 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} > mip360_bicycle.log &
sleep 20s

./train.sh -d mipnerf360/bonsai -l ${exp_name}  --gpu 1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} > mip360_bonsai.log &
sleep 20s

./train.sh -d mipnerf360/counter -l ${exp_name}  --gpu 2 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} > mip360_counter.log &
sleep 20s

./train.sh -d mipnerf360/garden -l ${exp_name}  --gpu 3 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} > mip360_garden.log &
sleep 20s

./train.sh -d mipnerf360/kitchen -l ${exp_name}  --gpu 4 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} > mip360_kitchen.log &
sleep 20s

./train.sh -d mipnerf360/room -l ${exp_name}  --gpu 5 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} > mip360_room.log &
sleep 20s

./train.sh -d mipnerf360/stump -l ${exp_name}  --gpu 6 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} > mip360_stump.log &
