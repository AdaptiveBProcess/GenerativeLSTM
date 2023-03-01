# DeepGenerator: Learning Accurate Generative Models of Business Processes with LSTM Neural Networks

The code here presented is able to execute different pre- and post-processing methods and architectures for building and using generative models from event logs in XES format using LSTM anf GRU neural networks. This code can perform the next tasks:


* Training LSTM neuronal networks using an event log as input.
* Generate full event logs using a trained LSTM neuronal network.
* Predict the remaining time and the continuation (suffix) of an incomplete business process trace. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

```
git clone https://github.com/AdaptiveBProcess/GenerativeLSTM.git
```

### Prerequisites

To execute this code you just need to install Anaconda in your system, and create an environment using the *environment.yml* specification provided in the repository.
```
cd GenerativeLSTM
conda env create -f environment.yml
conda activate deep_generator
```

## Running the script

Once created the environment, you can perform each one of the tasks, specifying the following parameters in the lstm.py module, or by command line as is described below:

*Training LSTM neuronal network:* To perform this task you need to set the required activity (-a) as 'training' followed by the name of the (-f) event log, and all the following parameters:

* Filename (-f): Log filename.
* Model family (-m): The available options are lstm, gru, lstm_cx and gru_cx.
* Max Eval (-e): Maximum number of evaluations.
* Opt method (-o): Optimization method used. The available options are hpc and bayesian.

```
(lstm_env) C:\sc_lstm>python dg_training.py -f Helpdesk.xes -m lstm -e 1 -o bayesian
```

*Predictive task:* It is possible to execute various predictive tasks with DeepGenerator, such as predicting the next event, the case continuation, and the remaining time of an ongoing case. Similarly, it is possible to generate complete event logs starting from a zero prefix size. To perform these tasks, you need to set the activity (-a) as ‘predict_next’ for the next event prediction, ‘pred_sfx’ for case continuation and remaining time, and ‘pred_log’ for the full event log generation. Additionally, it's required to indicate the folder where the predictive model is located (-c), and the name of the .h5 model (-b). Finally, you need to specify the method for selecting the next predicted task (-v) ‘random_choice’ or ‘arg_max’ and the number of repetitions of the experiment (-r). **NB! The folders and models were generated in the training task and can be found in the output_files folder:

```
(lstm_env) C:\sc_lstm>-a pred_log -c 20201001_426975C9_FAC6_453A_9F0B_4DD528CB554B -b "model_shared_cat_02-1.10.h5" -v "random_choice" -r 1"
```
*Predict the next event and role:* To perform this task the only changes with respect with the previous ones are that you need to set the required activity as 'predict_next' and its not necesary to set the maximum trace length:

```
(lstm_env) C:\sc_lstm>python lstm.py -a predict_next -c 20190228_155935509575 -b "model_rd_150 Nadam_22-0.59.h5" -x False
```
## Examples

Models examples and experimental results can be found at <a href="http://kodu.ut.ee/~chavez85/bpm2019/" target="_blank">examples</a>
## Authors

* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
