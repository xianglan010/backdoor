#### B1 Easy
#nohup python /workspaces/icsebowen26/repos/backdoor/run_codebert.py --train_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor1/0.01/seq2seq_easy/train.tsv --dev_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor1/0.01/seq2seq_easy/valid.tsv --output_dir /workspaces/icsebowen26/data/backdoor1/easy --cache_dir /workspaces/icsebowen26/datax/hugging_face_cache > /workspaces/icsebowen26/repos/backdoor/scripts/logs/easy.txt &
#### B1 Ambiguous
#nohup python /workspaces/icsebowen26/repos/backdoor/run_codebert.py --train_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor1/0.01/seq2seq_amb/train.tsv --dev_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor1/0.01/seq2seq_amb/valid.tsv --output_dir /workspaces/icsebowen26/data/backdoor1/amb --cache_dir /workspaces/icsebowen26/datax/hugging_face_cache > /workspaces/icsebowen26/repos/backdoor/scripts/logs/amb.txt &
#### B1 Hard
#nohup python /workspaces/icsebowen26/repos/backdoor/run_codebert.py --train_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor1/0.01/seq2seq_hard/train.tsv --dev_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor1/0.01/seq2seq_hard/valid.tsv --output_dir /workspaces/icsebowen26/data/backdoor1/hard --cache_dir /workspaces/icsebowen26/datax/hugging_face_cache > /workspaces/icsebowen26/repos/backdoor/scripts/logs/hard.txt &




#### B3 Easy
nohup python /workspaces/icsebowen26/repos/backdoor/run_codebert.py --train_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor3/0.01/seq2seq_easy/train.tsv --dev_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor3/0.01/seq2seq_easy/valid.tsv --output_dir /workspaces/icsebowen26/data/backdoor3/easy --cache_dir /workspaces/icsebowen26/datax/hugging_face_cache > /workspaces/icsebowen26/repos/backdoor/scripts/logs/easy_b3.txt &
#### B3 Ambiguous
#nohup python /workspaces/icsebowen26/repos/backdoor/run_codebert.py --train_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor3/0.01/seq2seq_amb/train.tsv --dev_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor3/0.01/seq2seq_amb/valid.tsv --output_dir /workspaces/icsebowen26/data/backdoor3/amb --cache_dir /workspaces/icsebowen26/datax/hugging_face_cache > /workspaces/icsebowen26/repos/backdoor/scripts/logs/amb_b3.txt &
#### B3 Hard
#nohup python /workspaces/icsebowen26/repos/backdoor/run_codebert.py --train_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor3/0.01/seq2seq_hard/train.tsv --dev_filename /workspaces/icsebowen26/semeru-datasets/security/backdoors/backdoor3/0.01/seq2seq_hard/valid.tsv --output_dir /workspaces/icsebowen26/data/backdoor3/hard --cache_dir /workspaces/icsebowen26/datax/hugging_face_cache > /workspaces/icsebowen26/repos/backdoor/scripts/logs/hard_b3.txt &