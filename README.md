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
https://drive.google.com/file/d/1RWStBN5zXEbjqSGDT8bQU38GHSo6q-P4/view?usp=sharing

Our dealt data and model checkpoints:<br>
Data: https://drive.google.com/file/d/1HG4fqg5vRh1KuMFp_aMNyZvh-9kEnlGb/view?usp=sharing <br>
Model: https://drive.google.com/file/d/1bcK4vP6xRc-1Xwv_Bkyz54t8XmKKg-ch/view?usp=sharing


## Usage

**In the development, we assign the "Diffusion" to the topic_module and the "Conformity" to the pop_module.**


### Prepare Data
[//]: # (```sh)
[//]: # (python main.py --phase get_model_graph_data --data_source geography --model DDHGCN)
[//]: # (python main.py --phase get_model_graph_data --data_source geography --model DPPDCC)
[//]: # (```)

```sh
python .\data_processor.py --phase make_data_graph --data_source geography
python main.py --phase get_model_graph_data --data_source geography --model DDHGCNSCL
python main.py --phase get_model_graph_data --data_source geography --model DPPDCC
```

### Training
```sh
# DPPDCC
python main.py --phase DPPDCC --data_source geography
```

### Testing

```sh
# DPPDCC
python main.py --phase test_results --model  DPPDCC --data_source geography
```
