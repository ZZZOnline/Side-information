
m='ml-100k'
y='yelp'
b='beer'
c='Amazon_Beauty'
a='ml-1m'
t='taobao'
r='randomijcai'
# i='ijcai'
# s='steam'

n='dim'

# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --hidden_units=64 --gpu_id=8>>dimout/s8.out&

# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=1 --lr=0.001 --cuda=11>>11.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=3 --lr=0.001 --cuda=10>>10.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=5 --lr=0.001 --cuda=9>>9.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=10 --lr=0.001 --cuda=8>>8.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=7>>7.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=30 --lr=0.001 --cuda=6>>6.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=50 --lr=0.001 --cuda=5>>5.out&

# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --gamma=0 --beam_size=20 --lr=0.001 --cuda=0>>0.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --gamma=0.5 --beam_size=20 --lr=0.001 --cuda=1>>1.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --gamma=1 --beam_size=20 --lr=0.001 --cuda=2>>2.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --gamma=3 --beam_size=20 --lr=0.001 --cuda=3>>3.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --gamma=5 --beam_size=20 --lr=0.001 --cuda=4>>4.out&

# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --gamma=0 --beam_size=20 --lr=0.001 --cuda=0&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --gamma=0.5 --beam_size=20 --lr=0.001 --cuda=1&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --gamma=1 --beam_size=20 --lr=0.001 --cuda=2&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --gamma=3 --beam_size=20 --lr=0.001 --cuda=3&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --gamma=5 --beam_size=20 --lr=0.001 --cuda=4&


# nohup python newcasrec.py --datasets=${t} --hidden_units=128 --gamma=1 --beam_size=20 --lr=0.001 --cuda=5&
# nohup python newcasrec.py --datasets=${r} --hidden_units=128 --gamma=1 --beam_size=20 --lr=0.001 --cuda=5&

# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=1 --lr=0.001 --cuda=11>>011.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=3 --lr=0.001 --cuda=10>>010.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=5 --lr=0.001 --cuda=4>>4.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=10 --lr=0.001 --cuda=3>>3.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=2>>2.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=30 --lr=0.001 --cuda=1>>1.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=50 --lr=0.001 --cuda=0>>0.out&

# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --gamma=0 --beam_size=20 --lr=0.001 --cuda=0>>0.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --gamma=0.5 --beam_size=20 --lr=0.001 --cuda=1>>1.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --gamma=1 --beam_size=20 --lr=0.001 --cuda=2>>2.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --gamma=3 --beam_size=20 --lr=0.001 --cuda=3>>3.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --gamma=5 --beam_size=20 --lr=0.001 --cuda=4>>4.out&


# nohup python newcasrec_mul2_no2.py --datasets=${b} --hidden_units=64 --beam_size=10 --lr=0.001 --cuda=0>>0.out&
# nohup python newcasrec_mul2_no1.py --datasets=${b} --hidden_units=64 --beam_size=10 --lr=0.001 --cuda=1>>1.out&


# nohup python newcasrec_mul2_no2.py --datasets=${t} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=11>>11.out&
# nohup python newcasrec_mul2_no1.py --datasets=${t} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=10>>10.out&
# nohup python newcasrec_mul2_no2.py --datasets=${r} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=9>>9.out&
# nohup python newcasrec_mul2_no1.py --datasets=${r} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=8>>8.out&


# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=32 --beam_size=20 --lr=0.001 --cuda=0>>00.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=64 --beam_size=20 --lr=0.001 --cuda=1>>01.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=2>>02.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=256 --beam_size=20 --lr=0.001 --cuda=4>>04.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=512 --beam_size=20 --lr=0.001 --cuda=3>>03.out&

# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=32 --beam_size=20 --lr=0.001 --cuda=10>>010.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=64 --beam_size=20 --lr=0.001 --cuda=11>>011.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=7>>07.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=256 --beam_size=20 --lr=0.001 --cuda=8>>08.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=512 --beam_size=20 --lr=0.001 --cuda=9>>09.out&

#
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=32 --beam_size=10 --lr=0.001 --cuda=1>>1.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=10 --lr=0.001 --cuda=0>>0.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=128 --beam_size=10 --lr=0.001 --cuda=2>>2.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=256 --beam_size=10 --lr=0.001 --cuda=3>>3.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=512 --beam_size=10 --lr=0.001 --cuda=4>>4.out&


# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=1 --lr=0.001 --cuda=11&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=3 --lr=0.001 --cuda=10&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=5 --lr=0.001 --cuda=9&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=10 --lr=0.001 --cuda=8&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=20 --lr=0.001 --cuda=7&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=30 --lr=0.001 --cuda=6&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=50 --lr=0.001 --cuda=5&


# nohup python newcasrec_mul2_hidden.py --datasets=${b} --hidden_units=64 --preference=add --beam_size=10 --lr=0.001 --cuda=11&
# nohup python newcasrec_mul2_hidden.py --datasets=${b} --hidden_units=64 --preference=dynamic --beam_size=10 --lr=0.001 --cuda=10&
# nohup python newcasrec_mul2_hidden.py --datasets=${b} --hidden_units=64 --preference=direct --beam_size=10 --lr=0.001 --cuda=9&


# nohup python newcasrec_mul2_inte.py --datasets=${b} --hidden_units=64 --integra=add --beam_size=10 --lr=0.001 --cuda=1>>01.out&
# nohup python newcasrec_mul2_inte.py --datasets=${b} --hidden_units=64 --integra=concat --beam_size=10 --lr=0.001 --cuda=2>>02.out&
# nohup python newcasrec_mul2_inte.py --datasets=${b} --hidden_units=64 --integra=item --beam_size=10 --lr=0.001 --cuda=3>>03.out&

# nohup python newcasrec_mul2_inte.py --datasets=${t} --hidden_units=128 --integra=add --beam_size=20 --lr=0.001 --cuda=0>>90.out&
# nohup python newcasrec_mul2_inte.py --datasets=${t} --hidden_units=128 --integra=concat --beam_size=20 --lr=0.001 --cuda=1>>91.out&
# nohup python newcasrec_mul2_inte.py --datasets=${t} --hidden_units=128 --integra=item --beam_size=20 --lr=0.001 --cuda=2>>92.out&

# nohup python newcasrec_mul2_inte.py --datasets=${r} --hidden_units=128 --integra=add --beam_size=20 --lr=0.001 --cuda=3>>93.out&
# nohup python newcasrec_mul2_inte.py --datasets=${r} --hidden_units=128 --integra=concat --beam_size=20 --lr=0.001 --cuda=4>>94.out&
# nohup python newcasrec_mul2_inte.py --datasets=${r} --hidden_units=128 --integra=item --beam_size=20 --lr=0.001 --cuda=8>>98.out&

# nohup python newcasrec_mul2_hidden.py --datasets=${b} --hidden_units=64 --preference=add --beam_size=10 --lr=0.001 --cuda=11>>11.out&
# nohup python newcasrec_mul2_hidden.py --datasets=${b} --hidden_units=64 --preference=dynamic --beam_size=10 --lr=0.001 --cuda=10>>10.out&
# nohup python newcasrec_mul2_hidden.py --datasets=${b} --hidden_units=64 --preference=direct --beam_size=10 --lr=0.001 --cuda=9>>9.out&
# nohup python newcasrec_mul2_hidden.py --datasets=${b} --hidden_units=64 --preference=gating --beam_size=10 --lr=0.001 --cuda=8>>8.out&


# nohup python newcasrec_mul2_hidden.py --datasets=${r} --hidden_units=128 --preference=add --beam_size=20 --lr=0.001 --cuda=11>>11.out&
# nohup python newcasrec_mul2_hidden.py --datasets=${r} --hidden_units=128 --preference=dynamic --beam_size=20 --lr=0.001 --cuda=10>>10.out&
# nohup python newcasrec_mul2_hidden.py --datasets=${r} --hidden_units=128 --preference=direct --beam_size=20 --lr=0.001 --cuda=9>>9.out&

# nohup python newcasrec_mul2_hidden.py --datasets=${t} --hidden_units=128 --preference=add --beam_size=20 --lr=0.001 --cuda=5>>5.out&
# nohup python newcasrec_mul2_hidden.py --datasets=${t} --hidden_units=128 --preference=dynamic --beam_size=20 --lr=0.001 --cuda=6>>6.out&
# nohup python newcasrec_mul2_hidden.py --datasets=${t} --hidden_units=128 --preference=direct --beam_size=20 --lr=0.001 --cuda=7>>7.out&

# nohup python newcasrec_mul2_hidden.py --datasets=${r} --hidden_units=128 --preference=add --beam_size=10 --lr=0.001 --cuda=11&
# nohup python newcasrec_mul2_hidden.py --datasets=${r} --hidden_units=128 --preference=dynamic --beam_size=10 --lr=0.001 --cuda=10&
# nohup python newcasrec_mul2_hidden.py --datasets=${r} --hidden_units=128 --preference=direct --beam_size=10 --lr=0.001 --cuda=9&

# nohup python newcasrec_mul2_hidden.py --datasets=${t} --hidden_units=128 --preference=add --beam_size=10 --lr=0.001 --cuda=8&
# nohup python newcasrec_mul2_hidden.py --datasets=${t} --hidden_units=128 --preference=dynamic --beam_size=10 --lr=0.001 --cuda=6&
# nohup python newcasrec_mul2_hidden.py --datasets=${t} --hidden_units=128 --preference=direct --beam_size=10 --lr=0.001 --cuda=7&

# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=10 --lr=0.001 --cuda=6>>6.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=10 --lr=0.001 --cuda=5>>5.out&
# nohup python newcasrec.py --datasets=taobao --gamma=1 --lr=0.001 --cuda=2>>32.out&
# nohup python newcasrec.py --datasets=randomijcai --gamma=1 --lr=0.001 --cuda=3>>33.out&
# nohup python newcasrec.py --datasets=taobao --gamma=3 --lr=0.001 --cuda=2>>30.out&
# nohup python newcasrec.py --datasets=randomijcai --gamma=3 --lr=0.001 --cuda=3>>30.out&

# nohup python newcasrec_mul2_hyper.py --datasets=beer --hidden_units=64 --beam_size=10 --gamma=0 --lr=0.001 --cuda=4&
# nohup python newcasrec_mul2_hyper.py --datasets=beer --hidden_units=64 --beam_size=10 --gamma=0.5 --lr=0.001 --cuda=7&
# nohup python newcasrec_mul2_hyper.py --datasets=beer --hidden_units=64 --beam_size=10 --gamma=1 --lr=0.001 --cuda=5&
# # nohup python newcasrec_mul2_hyper.py --datasets=beer --hidden_units=64 --beam_size=10 --gamma=3 --lr=0.001 --cuda=8&
# nohup python newcasrec_mul2_hyper.py --datasets=beer --hidden_units=64 --beam_size=10 --gamma=3 --lr=0.001 --cuda=9&
# nohup python newcasrec_mul2_hyper.py --datasets=beer --hidden_units=64 --beam_size=10 --gamma=3 --lr=0.001 --cuda=10&
# nohup python newcasrec_mul2_hyper.py --datasets=beer --hidden_units=64 --beam_size=10 --gamma=5 --lr=0.001 --cuda=5&

# nohup python newcasrec_mul2_no.py --datasets=beer --hidden_units=64 --beam_size=10 --gamma=1 --lr=0.001 --cuda6&
# nohup python newcasrec_mul2_no.py --datasets=${t} --hidden_units=128 --beam_size=20 --gamma=1 --lr=0.001 --cuda=7&
# nohup python newcasrec_mul2_no.py --datasets=${r} --hidden_units=128 --beam_size=20 --gamma=1 --lr=0.001 --cuda=8&

nohup python newcasrec_mul2_nocpe.py --datasets=beer --hidden_units=64 --beam_size=10 --gamma=1 --lr=0.001 --cuda6>>6.out&
nohup python newcasrec_mul2_nocpe.py --datasets=${t} --hidden_units=128 --beam_size=20 --gamma=1 --lr=0.001 --cuda=7>>7.out&
nohup python newcasrec_mul2_nocpe.py --datasets=${r} --hidden_units=128 --beam_size=20 --gamma=1 --lr=0.001 --cuda=8>>8.out&