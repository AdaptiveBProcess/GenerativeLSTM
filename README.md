# Learning Accurate Generative Models of Business Processes with LSTM Neural Networks

The code here presented is able to execute different pre- and post-processing methods and architectures for building and using generative models from event logs in XES format using LSTM neural networks. This code can perform the next tasks:


* Training embedded matrices for the activities and roles contained in an event log.
* Training LSTM neuronal networks using an event log as input.
* Generate full event logs using a trained LSTM neuronal network.
* Predict the remaining time and the continuation (suffix) of an incomplete business process trace. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

To execute this code you just need to install Anaconda in your system, and create an environment using the *lstm_env.yml* specification provided in the repository.

## Running the script

Once created the environment you can perform each one of the tasks, specifying the next parameters in the lstm.py module, or by command line as is described below:

*Training embedded matrices:* this task is a pre-requisite to train any LSTM model, to execute it you need to specify the required activity (-a) as 'emb_training' followed by the name of the event log (-e):

```
(lstm_env) C:\sc_lstm>python lstm.py -a emb_training -e Helpdesk.xes.gz
```
*Training LSTM neuronal network:* To perform this task you need to set the required activity as 'training' followed by the name of the event log, and all the following parameters:

* Implementation (-i): type of keras lstm implementation 1 cpu, 2 gpu
* lSTM activation function (-l): lSTM optimization function (see keras doc), None to set it up as the default value.
* Dense activation function (-d): dense layer activation function (see keras doc), None to set it up as the default value.
* optimization function (-o): optimization function (see keras doc).
* Scaling method (-n) = relative time between events scaling method max or lognorm.
* Model type (-t): type of LSTM model specialized, concatenated, or shared_cat.
* N-gram size (-b): Size of the n-gram (temporal dimension)
* LSTM layer sizes (-c): Size of the LSTM layers.

```
(lstm_env) C:\sc_lstm>python lstm.py -a training -e Helpdesk.xes.gz -i 1 -l None -d linear -o Nadam -n lognorm -t shared_cat -b 5 -c 100
```

*Generate full event log:* To perform this task you need to set the required activity as 'pred_log' followed by the folder (-f) and model (-m) names to be used to generate the event logs. This folders and models were generated tn the training task and can be found in the output_files folder:

```
(lstm_env) C:\sc_lstm>python lstm.py -a pred_log -f 20190228_155935509575 -m "model_rd_150 Nadam_22-0.59.h5"
```

*Predict the remaining time and suffix:* To perform this task you need to set the required activity as 'pred_sfx' followed by the folder (-f) and model (-m)names to be used to generate the event logs:

```
(lstm_env) C:\sc_lstm>python lstm.py -a pred_sfx -f 20190228_155935509575 -m "model_rd_150 Nadam_22-0.59.h5"
```

## Examples

Models examples and experimental results can be found at <a href="http://kodu.ut.ee/~chavez85/bpm2019/" target="_blank">examples</a>
## Authors

* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
