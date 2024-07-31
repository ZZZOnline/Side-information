m='ml-100k'
y='yelp'
b='beer'
c='Amazon_Beauty'
a='ml-1m'
t='taobao'
i='ijcai'
s='steam'

n='lr'

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0. --gpu_id=4>>normout/s8.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.01  --gpu_id=5>>normout/s9.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.05  --gpu_id=6>>normout/s10.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.1  --gpu_id=7>>normout/s11.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --gpu_id=8>>normout/s1.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=1  --gpu_id=9>>normout/s2.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=item --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=1  --gpu_id=10>>normout/s3.out&
nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2 --weight_norm=0.5 --lr=0.002 --gpu_id=0>>lrout/0a0.out&
nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2  --weight_norm=0.5 --lr=0.0001 --gpu_id=1>>lrout/0a1.out&
nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2  --weight_norm=0.5 --lr=0.0002 --gpu_id=2>>lrout/0a2.out&