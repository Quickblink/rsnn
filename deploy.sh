docker build -t imrsnn . \
&& nvidia-docker run \
-v ~/eric/rsnn:/home/developer \
--name rsnn_co \
--net host \
--ipc host \
--rm \
-t \
imrsnn \
python3 Code/run_exp.py
