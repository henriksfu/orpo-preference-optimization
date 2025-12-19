all: train dev zipout check

train:
	python3 answer/train_model.py

dev:
	python3 answer/prefopt.py -i data/input/dev.txt -d cuda:0 -l log_dev.txt > output_dev.txt
	python3 output_check.py -t data/reference/dev.out -o output_dev.txt

zipout:
	python3 zipout.py

check: 
	python3 check.py