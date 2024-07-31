m='ml-100k'
y='yelp'
b='beer'
c='Amazon_Beauty'
a='ml-1m'
t='taobao'
i='ijcai'
s='steam'

n='dim'

# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=64 --gpu_id=8>>dimout/s8.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=128 --gpu_id=9>>dimout/s9.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=256 --gpu_id=10>>dimout/s10.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=512 --gpu_id=11>>dimout/s11.out&

# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=64 --gpu_id=10>>dimout/a10.out&
# # nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=128 --gpu_id=9>>dimout/s9.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=256 --gpu_id=11>>dimout/a11.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=512 --gpu_id=3>>dimout/a3.out&

nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=64 --gpu_id=8>>dimout/s8.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=128 --gpu_id=9>>dimout/s9.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=256 --gpu_id=10>>dimout/s10.out&