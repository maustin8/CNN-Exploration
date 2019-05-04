# CNN-Exploration

This is a machine learning library developed by Max Austin for CS5350/6350 in University of Utah

Note: the tensorflow library is required to run

# To run program:  
Commands:  
Optional </br>

"-a", "--model_arch":Choose the CNN architecture to use {mnist, alex, vgg} [DEFAULT = mnist] </br>
"-d", "--datset_direc":Choose the directory for the data [DEFAULT = dataset/test/] </br>
"-m", "--model_direc":Choose the directory to save the trained model to [DEFAULT = saved_models/] </br>
"-c", "--check_direc":Choose the directory to save the training checkpoints to [DEFAULT = training_checkpoints/] </br>
"-p", "--check_period":Choose how often to save a checkpoint [DEFAULT = 5] </br>
"-e", "--num_epoch":Choose the number of epochs for training (Note: If multiple values are given, a model will be trained with each) [DEFAULT = [5]] </br>
"-b", "--batch_size":Choose the batch size for training (Note: If multiple values are given, a model will be trained with each) [DEFAULT = [5]] </br>
"-n", "--model_name":Choose the name to save the trained model as [DEFAULT] = e[num_epoch]_s_[epoch_step]_model_latest.h5] </br>
"-l", "--load_model":Load trained h5 model </br>
"-r", "--resume":Resume from specified checkpoint </br>
"-v", "--verbose":Flag to write evaluation results to the file results_cnn.txt Note: This file will be overwritten </br>
example: </br>
python main.py -e 25 50 100 150 -b 5 10 50 100 200 -d dataset/area/ -a alex -v