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
            
            model_logs.append({'loss':data['loss'], 'acc':data['accuracy']})
            data.close()
        
    #model_logs = np.array(model_logs)
        
    fig, axs = plt.subplots(1, 2, figsize=(5.50107, 9.00177/7))
    axs = axs.ravel()
    
    for log in model_logs:
        batches = np.arange(0, len(log['acc']))
        axs[0].plot(batches, log['loss'], c= u'#1f77b4', alpha = 0.2)
    
        axs[1].plot(batches, log['acc'], c= u'#ff7f0e', alpha = 0.2)
    
    
    length = max([len(log['acc']) for log in model_logs])
    
    #mean_loss, mean_acc = [], []
    #loss = [0 for i in range(len(model_logs))]
    #acc  = [0 for i in range(len(model_logs))]
    #for t in range(length):        
    #    for i in range(len(model_logs)):
    #        try:
    #            loss[i] = model_logs[i]['loss'][t,0]
    #            acc[i]  = model_logs[i]['acc'][t,0]
    #        except IndexError:
    #            pass
    #    
    #    mean_loss.append(sum(loss)/len(loss))
    #    mean_acc.append(sum(acc)/len(acc))
    
    batches = np.arange(0, length)
    
    axs[0].set_xlabel('Batch')
    axs[0].set_ylabel('CE-Loss')
    #axs[0].plot(batches, model_logs_means[0,:], c= u'#1f77b4', label='Mean Loss')
    axs[0].plot(batches[49:-50], movingaverage(log['loss'], 100), 
                c= u'#1f77b4', ls='-', label='(Smoothed) Loss')
    axs[0].set_xlim([0,7500])
    
    axs[1].set_xlabel('Batch')
    axs[1].set_ylabel('Accuracy')
    #axs[1].plot(batches, model_logs_means[1,:], c= u'#ff7f0e',  label='Mean Acc')
    axs[1].plot(batches[2:-2], movingaverage(log['acc'], 5), 
                c= u'#ff7f0e', ls='-', label='(Smoothed) Acc')
    axs[1].set_xlim([0,7500])
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

loss_plots('BartlebytheScrivener_Train_Losses', 'Bartleby the Scrivener')
loss_plots('PyDocTutorials_Train_Losses.npz', 'PyDoc Tutorials')
