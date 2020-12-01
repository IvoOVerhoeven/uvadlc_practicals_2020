"""Generating plots."""
import matplotlib.pyplot as plt
import numpy as np
import os

def loss_plots(experiment_id, title=None):
    
    if title is None: title = experiment_id
    
    def movingaverage(data, window_size):
        """ 
        Function for computing a smoothed, moving average, for the losses.
        Taken from https://stackoverflow.com/a/11352216/11692721
        """
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(data, window, 'valid')
    
    model_logs = []
    for model_log in os.listdir('./models'):
        if model_log.startswith(experiment_id):
            data = np.load('./models/'+ model_log)
            model_logs.append({'loss':data['loss'], 'acc':data['acc']})
            data.close()
        
    #model_logs = np.array(model_logs)
        
    fig, axs = plt.subplots(1, 2, figsize=(5.50107, 9.00177/7))
    axs = axs.ravel()
    
    for log in model_logs:
        batches = np.arange(0, len(log['acc']))
        axs[0].plot(batches, log['loss'][:, 0], c= u'#1f77b4', alpha = 0.2)
    
        axs[1].plot(batches, log['acc'][:, 0], c= u'#ff7f0e', alpha = 0.2)
    
    
    length = max([len(log['acc']) for log in model_logs])
    
    mean_loss, mean_acc = [], []
    loss = [0 for i in range(len(model_logs))]
    acc  = [0 for i in range(len(model_logs))]
    for t in range(length):        
        for i in range(len(model_logs)):
            try:
                loss[i] = model_logs[i]['loss'][t,0]
                acc[i]  = model_logs[i]['acc'][t,0]
            except IndexError:
                pass
        
        mean_loss.append(sum(loss)/len(loss))
        mean_acc.append(sum(acc)/len(acc))
    
    batches = np.arange(0, length)
    
    axs[0].set_xlabel('Batch')
    axs[0].set_ylabel('CE-Loss')
    #axs[0].plot(batches, model_logs_means[0,:], c= u'#1f77b4', label='Mean Loss')
    axs[0].plot(batches[2:-2], movingaverage(mean_loss, 5), 
                c= u'#1f77b4', ls='-', label='(Smoothed) Mean Loss')
    axs[0].set_xlim([0,2000])
    
    axs[1].set_xlabel('Batch')
    axs[1].set_ylabel('Accuracy')
    #axs[1].plot(batches, model_logs_means[1,:], c= u'#ff7f0e',  label='Mean Acc')
    axs[1].plot(batches[2:-2], movingaverage(mean_acc, 5), 
                c= u'#ff7f0e', ls='-', label='(Smoothed) Mean Acc')
    axs[1].set_xlim([0,2000])
    axs[1].set_ylim([-0.05,1.05])
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    
    fig.suptitle(title, fontsize=11, y=1.01)
    #fig.legend(bbox_to_anchor=(1, 0.9), loc='upper left')
    
    fig.savefig(os.path.join('./figures/' + title + '.pdf'), 
                bbox_inches='tight')
    
    return fig

##############################################################################
### Plot Calls
##############################################################################

# %matplotlib inline

loss_plots('LSTM-Seq_length%3A5-Input_dim%3A1', 'LSTM, T=5, d=1')
loss_plots('LSTM-Seq_length%3A10-Input_dim%3A1', 'LSTM, T=10, d=1')
loss_plots('LSTM-Seq_length%3A15-Input_dim%3A1', 'LSTM, T=15, d=1')

loss_plots('biLSTM-Seq_length%3A5-Input_dim%3A1', 'Bi-LSTM, T=5, d=1')
loss_plots('biLSTM-Seq_length%3A10-Input_dim%3A1', 'Bi-LSTM, T=10, d=1')
loss_plots('biLSTM-Seq_length%3A15-Input_dim%3A1', 'Bi-LSTM, T=15, d=1')

loss_plots('LSTM-Seq_length%3A5-Input_dim%3A10', 'LSTM, T=5, d=10')
loss_plots('LSTM-Seq_length%3A10-Input_dim%3A20', 'LSTM, T=10, d=20')
loss_plots('LSTM-Seq_length%3A15-Input_dim%3A30', 'LSTM, T=15, d=30')

loss_plots('biLSTM-Seq_length%3A5-Input_dim%3A10', 'Bi-LSTM, T=5, d=10')
loss_plots('biLSTM-Seq_length%3A10-Input_dim%3A20', 'Bi-LSTM, T=10, d=20')
loss_plots('biLSTM-Seq_length%3A15-Input_dim%3A30', 'Bi-LSTM, T=15, d=30')

loss_plots('LSTM-Seq_length%3A5-Input_dim%3A64', 'LSTM, T=5, d=64')
loss_plots('LSTM-Seq_length%3A10-Input_dim%3A64', 'LSTM, T=10, d=64')
loss_plots('LSTM-Seq_length%3A15-Input_dim%3A64', 'LSTM, T=15, d=64')

loss_plots('biLSTM-Seq_length%3A5-Input_dim%3A64', 'Bi-LSTM, T=5, d=64')
loss_plots('biLSTM-Seq_length%3A10-Input_dim%3A64', 'Bi-LSTM, T=10, d=64')
loss_plots('biLSTM-Seq_length%3A15-Input_dim%3A64', 'Bi-LSTM, T=15, d=64')