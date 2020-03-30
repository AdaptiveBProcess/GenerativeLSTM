# Learning Accurate Generative Models of Business Processes with LSTM Neural Networks

The code here presented can execute different pre- and post-processing methods and architectures for building and using generative models from event logs in XES format using LSTM neural networks. This code can perform the next tasks:


* Training embedded matrices for the activities and roles contained in an event log.
* Training LSTM neuronal networks using an event log as input.
* Generate full event logs using a trained LSTM neuronal network.
* Predict the remaining time and the continuation (suffix) of an incomplete business process trace. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

To execute this code, you need to install Anaconda in your system and create an environment using the *lstm_env.yml* specification provided in the repository.

## Running the script

Once created the environment, you can perform each one of the tasks, specifying the following parameters in the lstm.py module, or by command line as is described below:

*Training embedded matrices:* this task is a pre-requisite to train an LSTM model, to execute it you need to specify the required activity (-a) as 'emb_training' followed by the name of the event log (-f), and define if the event-log has a single timestamp or not (-o):

```
(lstm_env) C:\sc_lstm>python lstm.py -a emb_training -f Helpdesk.xes.gz -o True
```
*Training LSTM neuronal network:* To perform this task, you need to set the required activity as 'training' followed by the name of the event log, and all the following parameters:

* One timestamp Event-log (-o): define if the event-log has a single timestamp or not
* Implementation (-i): type of Keras LSTM implementation 1 cpu, 2 gpu
* LSTM activation function (-l): LSTM optimization function (see Keras doc), None to set it up as the default value.
* Dense activation function (-d): dense layer activation function (see Keras doc), None to set it up as the default value.
* optimization function (-p): optimization function (see Keras doc).
* Scaling method (-n) = relative time between events scaling method max or lognorm.
* Model type (-m): type of LSTM model specialized, concatenated, or shared_cat.
* N-gram size (-z): Size of the n-gram (temporal dimension)
* LSTM layer sizes (-y): Size of the LSTM layers.

```
(lstm_env) C:\sc_lstm>python lstm.py -a training -f Helpdesk.xes -o True -i 1 -l sigmoid -d None -p Nadam -n max -m shared_cat -z 5 -y 50
```

*Generate full event log:* To perform this task, you need to set the required activity as 'pred_log' followed by the folder (-c) and model (-b) names to be used to generate the event logs. These folders and models were generated during the training task and can be found in the output_files folder. Additionally, you need to specify the maximum length of the predicted traces (-t). Finally, to store the results, it's necessary to define if you are executing the task as a single execution or if you are running other prediction instances (-x). If it's a single execution, the detailed results and individual measurements are stored in a subfolder called results. Otherwise, the results of all the running models are store in the output_files folder in a shared file:

```
(lstm_env) C:\sc_lstm>python lstm.py -a pred_log -c 20190228_155935509575 -b "model_rd_150 Nadam_22-0.59.h5" -t 100 -x False
```

*Predict the remaining time and suffix:* To perform this task, the only change with respect with the previous one is that you need to set the required activity as 'pred_sfx':

```
(lstm_env) C:\sc_lstm>python lstm.py -a pred_sfx -c 20190228_155935509575 -b "model_rd_150 Nadam_22-0.59.h5" -t 100 -x False
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
