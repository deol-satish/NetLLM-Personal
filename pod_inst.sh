ssh root@69.30.85.20 -p 22135 -i ~/.ssh/id_ed25519


ssh root@69.30.85.20 -p 22057 -i ~/.ssh/id_ed25519

chmod 600 ~/.ssh/id_ed25519;chmod 600 ~/.ssh/id_ed25519.pub


sudo scp -P 3322 -p -i ~/.ssh/mptcprootkey -r NetLLM-Personal root@192.168.56.1:
sudo scp -334423 -p -i ~/.ssh/mptcprootkey -r NetLtLLM-Personal root@192.168.56.1:

scp -P 22135 -i ~/.ssh/id_ed25519 -r NetLLM-Personal root@69.30.85.20:/path/to/destination/

ssh root@69.30.85.48 -p 22177 -i ~/.ssh/id_ed25519


sudo scp -P 22177 -p -i ~/.ssh/id_ed25519 -r NetLLM-Personal root@69.30.85.48:

ssh root@69.30.85.24 -p 22163 -i ~/.ssh/id_ed25519



python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --device-out cuda:1 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --device-out cuda:1 --lr 0.0001 --warmup-steps 1000 --num-epochs 20 --eval-per-epoch 2



Test performance of NetLLM:
python run_plm.py --test --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --device-out cuda:1 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2