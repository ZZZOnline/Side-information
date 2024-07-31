
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


# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=1 --lr=0.001 --cuda=11>>011.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=3 --lr=0.001 --cuda=10>>010.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=5 --lr=0.001 --cuda=4>>4.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=10 --lr=0.001 --cuda=3>>3.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=2>>2.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=30 --lr=0.001 --cuda=1>>1.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${r} --hidden_units=128 --beam_size=50 --lr=0.001 --cuda=0>>0.out&

# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=1 --lr=0.001 --cuda=11>>011.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=3 --lr=0.001 --cuda=10>>010.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=5 --lr=0.001 --cuda=4>>4.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=10 --lr=0.001 --cuda=3>>3.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=20 --lr=0.001 --cuda=2>>2.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=30 --lr=0.001 --cuda=1>>1.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=50 --lr=0.001 --cuda=0>>0.out&


# nohup python fdsa_mul.py --datasets=${r}  --lr=0.001 --fdsa_alpha=0.1 --cuda=11>>011.out&
# nohup python fdsa_mul.py --datasets=${r}  --lr=0.001 --fdsa_alpha=0.2 --cuda=10>>010.out&
# nohup python fdsa_mul.py --datasets=${r}  --lr=0.001 --fdsa_alpha=0.3 --cuda=1>>01.out&


# nohup python fdsa_mul.py --datasets=${b}  --lr=0.001 --fdsa_alpha=0.1 --cuda=0>>0.out&
# nohup python fdsa_mul.py --datasets=${b}  --lr=0.001 --fdsa_alpha=0.2 --cuda=2>>3.out&
# nohup python fdsa_mul.py --datasets=${b}  --lr=0.001 --fdsa_alpha=0.3 --cuda=3>>3.out&

# nohup python fdsa_mul.py --datasets=${t}  --lr=0.001 --fdsa_alpha=0.1 --cuda=5>>05.out&
# nohup python fdsa_mul.py --datasets=${t}  --lr=0.001 --fdsa_alpha=0.2 --cuda=4>>04.out&
# nohup python fdsa_mul.py --datasets=${t}  --lr=0.001 --fdsa_alpha=0.3 --cuda=6>>06.out&

# nohup python fdsa.py --datasets=${t}  --lr=0.00001 --cuda=11>>11.out&
# nohup python difsr.py --datasets=${t} --lr=0.00001 --cuda=10>>10.out&
# # nohup python nova.py --datasets=${t} --lr=0.001 --cuda=9>>9.out&
# nohup python gru4rec.py --datasets=${t} --lr=0.00001 --cuda=8>>8.out&
# # nohup python sas.py --datasets=${t} --lr=0.001 --cuda=7>>7.out&
# nohup python bert4rec.py --datasets=${t} --lr=0.001 --cuda=6>>6.out&
# nohup python caser.py --datasets=${t} --lr=0.001 --cuda=5>>5.out&


# nohup python fdsa.py --datasets=${r}  --lr=0.00001 --cuda=2>>2.out&
# nohup python difsr.py --datasets=${r} --lr=0.00001 --cuda=3>>3.out&
# nohup python nova.py --datasets=${r} --lr=0.001 --cuda=4>>9.out&
# nohup python gru4rec.py --datasets=${r} --lr=0.00001 --cuda=8>>8.out&
# nohup python sas.py --datasets=${r} --lr=0.001 --cuda=7>>7.out&
# nohup python bert4rec.py --datasets=${r} --lr=0.001 --cuda=6>>6.out&
# nohup python caser.py --datasets=${r} --lr=0.001 --cuda=5>>5.out&


# nohup python newcasrec_mul2_no2.py --datasets=${t} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=11>>11.out&
# nohup python newcasrec_mul2_no1.py --datasets=${t} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=10>>10.out&
# nohup python newcasrec_mul2_no2.py --datasets=${r} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=9>>9.out&
# nohup python newcasrec_mul2_no1.py --datasets=${r} --hidden_units=128 --beam_size=20 --lr=0.001 --cuda=8>>8.out&

# #
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=32 --beam_size=10 --lr=0.001 --cuda=1>>1.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=64 --beam_size=10 --lr=0.001 --cuda=0>>0.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=128 --beam_size=10 --lr=0.001 --cuda=2>>2.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=256 --beam_size=10 --lr=0.001 --cuda=3>>3.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${b} --hidden_units=512 --beam_size=10 --lr=0.001 --cuda=4>>4.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=10 --lr=0.001 --cuda=6>>6.out&
# nohup python newcasrec_mul2_hyper.py --datasets=${t} --hidden_units=128 --beam_size=10 --lr=0.001 --cuda=5>>5.out&
# nohup python newcasrec.py --datasets=taobao --gamma=1 --lr=0.001 --cuda=2>>32.out&
# nohup python newcasrec.py --datasets=randomijcai --gamma=1 --lr=0.001 --cuda=3>>33.out&
# nohup python newcasrec.py --datasets=taobao --gamma=3 --lr=0.001 --cuda=2>>30.out&
# nohup python newcasrec.py --datasets=randomijcai --gamma=3 --lr=0.001 --cuda=3>>30.out&

# nohup python newcasrec_mul2.py --datasets=beer --gamma=0 --lr=0.001 --cuda=4>>24.out&
# nohup python newcasrec_mul2.py --datasets=beer --gamma=0.5 --lr=0.001 --cuda=7>>27.out&
# nohup python newcasrec_mul2.py --datasets=beer --gamma=1 --lr=0.001 --cuda=5>>25.out&
# nohup python newcasrec_mul2.py --datasets=beer --gamma=3 --lr=0.001 --cuda=8>>28.out&
# nohup python newcasrec_mul2.py --datasets=beer --gamma=3 --lr=0.002 --cuda=9>>28.out&
# nohup python newcasrec_mul2.py --datasets=beer --gamma=3 --lr=0.002 --cuda=10>>28.out&
# nohup python newcasrec_mul2.py --datasets=beer --gamma=5 --lr=0.001 --cuda=6>>26.out&