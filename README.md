# DPPDCC
This is the official repo for "Predicting Scientific Impact Through Diffusion, Conformity, and Contribution Disentanglement".

The preprocessed dataset S2AG can be downloaded through this API:
https://www.semanticscholar.org/product/api

The code and repo will be further cleared up soon.

We also incorporate the supplementary material ([DPPDCC_supplementary_material.pdf](DPPDCC_supplementary_material.pdf)).


## Requirement
```sh
pip install -r requirements.txt
```
You can download demo dataset here:
https://drive.google.com/file/d/1rA0kr-VMY51hZIxplPcs-M6n6VJmzbwx/view?usp=sharing

We will share our dealt data and checkpoints here:
XXX


## Usage

**In the development, we assign the "Diffusion" to the topic_module and the "Conformity" to the pop_module.**


### Prepare Data
[//]: # (```sh)
[//]: # (python main.py --phase get_model_graph_data --data_source geography --model DDHGCN)
[//]: # (python main.py --phase get_model_graph_data --data_source geography --model DPPDCC)
[//]: # (```)

```sh
python main.py --phase get_model_graph_data --data_source geography --model DPPDCC
```

### Training
```sh
# DPPDCC
python main.py --phase DPPDCC --data_source geography --n_layers 4 --encoder_type 'CCompGATSM' --topic_module 'scl' --pop_module 'accum' --type orthog --num_workers 8
```

### Testing

```sh
# DPPDCC
python main.py --phase test_results --model  DPPDCC --data_source geography --n_layers 4 --encoder_type 'CCompGATSM' --topic_module 'scl' --pop_module 'accum' --type orthog --num_workers 8
```
