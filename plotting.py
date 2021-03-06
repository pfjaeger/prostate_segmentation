
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_batch_prediction(img, seg, prediction, num_classes, outfile, dim=2, n_select_from_batch=None):
    """
    plot the input image + ground truth segmentation + predictions for one batch
    :param img: shape [b, x, y, ch] (2D) / [b, z, x, y, ch] (3D)
    :param seg: shape [b, x, y, n_classes] (2D) / [b, z, x, y, n_classes] (3D)
    :param prediction: [b, x, y] (2D) / [b, z, x, y] (3D)
    """
    if dim == 3:
        seg = np.argmax(seg[0],axis=3)[:, :, :, np.newaxis]
        prediction = prediction[0, :, :, :, np.newaxis]
        img = img[0]
    else:
        seg = np.argmax(seg, axis=3)[:, :, :, np.newaxis]
        prediction = prediction[:, :, :, np.newaxis]


    try:
        # all dimensions except for the 'channel-dimension' are required to match
        for i in [0, 1, 2]:
            assert img.shape[i] == seg.shape[i] == prediction.shape[i]
    except:
        raise Warning('Shapes of arrays to plot not in agreement! Shapes {} vs. {} vs {}'.format(data.shape,
                                                                                                 seg.shape,
                                                                                             prediction.shape))

    show_arrays = np.concatenate([img, seg, prediction], axis=3)[:n_select_from_batch]
    fig, axarr = plt.subplots(show_arrays.shape[3], show_arrays.shape[0])
    fig.set_figwidth(2.5 * show_arrays.shape[0])
    fig.set_figheight(2.5 * show_arrays.shape[3])

    for b in range(show_arrays.shape[0]):
        for m in range(show_arrays.shape[3]):

            if m < img.shape[3]:
                cmap = 'gray'
                vmin = None
                vmax = None
            else:
                cmap = None
                vmin = 0
                vmax = num_classes - 1

            axarr[m, b].axis('off')
            axarr[m, b].imshow(show_arrays[b, :, :, m], cmap=cmap, vmin=vmin, vmax=vmax)

    plt.savefig(outfile)
    plt.close(fig)



class TrainingPlot_2Panel():

    def __init__(self,
                 num_epochs,
                 file_name,
                 experiment_name,
                 class_dict=None,
                 figsize = (10, 8), ymax=1):

        self.file_name = file_name
        self.exp_name = experiment_name
        self.class_dict = class_dict
        self.f = plt.figure(figsize=figsize)
        gs1 = gridspec.GridSpec(2, 1, height_ratios=[3.5,1], width_ratios=[1])
        self.ax1 = plt.subplot(gs1[0])
        self.ax2 = plt.subplot(gs1[1])

        self.ax1.set_xlabel('epochs')
        self.ax1.set_ylabel('loss / class dice coeffs.')
        self.ax1.set_xlim(0,num_epochs)
        self.ax1.set_ylim(0.0, ymax)
        # self.ax1.set_aspect(num_epochs//2)

        self.ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        self.ax2.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    def update_and_save(self,
                        metrics,
                        best_metrics,
                        type='loss_and_dice'):

        plot_loss_and_dice(self.ax1, self.ax2, metrics, best_metrics, self.exp_name, self.class_dict)
        plt.savefig(self.file_name)



def plot_loss_and_dice(ax, ax2, metrics, best_metrics, experiment_name, class_dict=None):
    """
    monitor the training process in terms of the loss and dice values of the individual classes
    """
    num_epochs = len(metrics['val']['loss'])
    epochs = range(num_epochs)
    num_classes = len(best_metrics['dices']) -1

    # prepare colors and linestyle
    num_lines = num_classes + 1
    color=iter(plt.cm.rainbow(np.linspace(0,1,num_lines)))
    colors = []
    for _ in range(num_lines):
        colors.append(next(color))

    colors = colors + colors
    linestyle = ['--']*num_lines + ['-']*num_lines

    # prepare values
    values_to_plot = []
    for l in range(num_classes):
        values_to_plot.append(metrics['train']['dices'][:,l])
    values_to_plot.append(metrics['train']['loss'])
    for l in range(num_classes):
        values_to_plot.append(metrics['val']['dices'][:,l])
    values_to_plot.append(metrics['val']['loss'])

    # prepare legend
    if class_dict != None:
        assert len(class_dict) == num_classes +1
        raw_labels = [class_dict[i] + ' dice' for i in range(num_classes)]
    else:
        raw_labels = ['class ' + str(i) + ' dice' for i in range(num_classes)]
    raw_labels.append('Loss')

    train_labels = ['Train: ' + l for l in raw_labels]
    val_labels = ['Val: ' + l for l in raw_labels]
    labels = train_labels + val_labels

    if ax.lines:
        for i, elem in enumerate(values_to_plot):
            ax.lines[i].set_xdata(epochs)
            ax.lines[i].set_ydata(elem)
    else:
        for elem, color, linestyle, label in zip(values_to_plot, colors, linestyle, labels):
            ax.plot(epochs, elem, color=color, linestyle=linestyle, label=label)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc=1, fontsize=10)
    leg.get_frame().set_alpha(0.5)
    text = "EXPERIMENT_NAME = '{}'\nBest Val Loss/Ep = {}/{}\n"\
        .format(experiment_name, np.round(best_metrics['loss'][0],3), best_metrics['loss'][1])

    best_metrics_text = ''
    for c in range(num_classes + 1):
        best_metrics_text += 'Best {}-Dice/Ep = {}/{}\n'.format(class_dict[c], np.round(best_metrics['dices'][c][0],3), int(best_metrics['dices'][c][1]))

    text += best_metrics_text
    ax2.clear()
    ax2.text(0.03, 0.1, text, color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0))



def plot_batch_gen_example(batch, cf, dim=2):
    """
    test the data generators by plotting example batches
    """
    if dim==3:
        img = batch['data'][0]
        seg = np.argmax(batch['seg'][0], axis=3)
    else:
        img = batch['data']
        seg = np.argmax(batch['seg'], axis=3)
    fig, axarr = plt.subplots(2, img.shape[0])
    fig.set_figheight(6)
    fig.set_figwidth(img.shape[0]*2.5)
    for b in range(axarr.shape[1]):
        axarr[0, b].imshow(img[b, :, :, 0], cmap='gray')
        axarr[0, b].axis('off')
        axarr[1, b].imshow(seg[b, :, :])
        axarr[1, b].axis('off')

    plt.savefig(cf.plot_dir + '/batch_example.png')
    plt.close(fig)