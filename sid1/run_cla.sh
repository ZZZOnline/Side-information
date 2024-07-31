m='ml-100k'
y='yelp'
b='beer'
c='Amazon_Beauty'
a='ml-1m'
t='taobao'
i='ijcai'
s='steam'

n='cla'

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=adamct_decode1 --weight_norm=0.1 --gpu_id=3>>claout/y3.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=gru4rec_decode --weight_norm=0.1  --gpu_id=4>>claout/y4.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=nova_decode --weight_norm=0.1  --gpu_id=5>>claout/y5.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=difsr_decode1 --weight_norm=0.1  --gpu_id=6>>claout/y6.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=mlp4rec_decode1 --weight_norm=0.1 --gpu_id=7>>claout/y07.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=fdsa_decode --weight_norm=0.1  --gpu_id=10>>claout/y10.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=mlp4rec --weight_norm=0.1 --gpu_id=5>>claout/y05.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=fdsa --weight_norm=0.1  --gpu_id=6>>claout/y6.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=bert4rec_decode --weight_norm=0.1  --gpu_id=11>>claout/y11.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore22 --weight_norm=0.1  --gpu_id=1>>claout/y0001.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore22both0 --weight_norm=0.1  --gpu_id=2>>claout/y0002.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2aw --weight_norm=0.1  --gpu_id=0>>claout/y0000.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2cv --weight_norm=0.1  --gpu_id=4>>claout/y0004.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2awcv --weight_norm=0.1  --gpu_id=4>>claout/y400.out&


# nohup python run_recbole_${n}.py --dataset=${y} --integra=item --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.1  --gpu_id=0>>claout/y000.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=item --pooling_mode=mean  --model=ours4_no --weight_norm=0.1  --gpu_id=2>>claout/y200.out&

nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2aw --weight_norm=0.5  --gpu_id=8>>claout/s8.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2cv --weight_norm=0.5  --gpu_id=9>>claout/s9.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2awcv --weight_norm=0.5  --gpu_id=10>>claout/s10.out&


# nohup python run_recbole_${n}.py --dataset=${y} --integra=item --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.1  --gpu_id=0>>claout/y000.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=item --pooling_mode=mean  --model=ours4_no --weight_norm=0.1  --gpu_id=2>>claout/y200.out&

