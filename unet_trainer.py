__author__ = 'Paul F. Jaeger'


import numpy as np
import tensorflow as tf
from plotting import plot_batch_prediction, TrainingPlot_2Panel
from utils import softmax_2d, get_dice_per_class, get_class_weights
from model import create_nice_UNet as create_UNet



class Unet(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=3, num_classes=2, cost="cross_entropy", cost_kwargs={}, **kwargs):
        tf.reset_default_graph()

        self.n_class = num_classes
        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder('float', shape=[None, 288, 288, channels])
        self.y = tf.placeholder('float', shape=[None, 288, 288, num_classes])
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.class_weights = tf.placeholder('float')

        logits, self.variables = create_UNet(self.x, num_classes=num_classes, is_training=self.is_training, **kwargs)

        self.cost = self._get_cost(logits, cost, cost_kwargs)
        self.logits = logits
        self.predicter = softmax_2d(logits)
        self.dice_per_class = get_dice_per_class(logits, self.y)


    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are: 
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])
        # USE FLAT ONES TO ACCESS NEW STANDARD SOFTMAX FUNCTIONS TAHT WORK MORE STABLE?
        if cost_name == "cross_entropy":

            if self.class_weights is not None:

                weight_map = tf.multiply(flat_labels, self.class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)

                loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                   labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)

                loss = tf.reduce_mean(weighted_loss)

            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                              labels=flat_labels))
        elif cost_name == "dice_coefficient":
            loss= tf.constant(1.) - get_dice_per_class(logits, self.y)[1]


        return loss




class Trainer(object):
    """
    Trains a unet instance

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

    """


    def __init__(self, net, cf, fold, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.cf = cf
        self.epoch = 0
        self.fold = fold
        self.metrics = {}
        self.metrics['train'] = {'loss': [0.], 'dices': np.zeros(shape=(1, self.cf.num_classes))}
        self.metrics['val'] = {'loss': [0.], 'dices': np.zeros(shape=(1, self.cf.num_classes))}
        self.best_metrics = {'loss': [10, 0], 'dices': np.zeros(shape=(self.cf.num_classes, 2))}
        file_name = self.cf.exp_dir + '/monitor_{}.png'.format(fold)
        self.TrainingPlot = TrainingPlot_2Panel(self.cf.num_epochs, file_name, self.cf.experiment_name,
                                                self.cf.class_dict)


    def _get_optimizer(self, global_step):


        learning_rate = self.opt_kwargs.pop("learning_rate", 0.00001)
        self.learning_rate_node = tf.Variable(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print "CHECK UPDATE OPS:", update_ops
        if update_ops:
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                           global_step=global_step)

        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                           global_step=global_step)

        return optimizer

    def _initialize(self, training_iters):
        global_step = tf.Variable(0)


        self.optimizer = self._get_optimizer(global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)


        init = tf.global_variables_initializer()

        return init



    def train(self, batch_gen):

        init = self._initialize(self.cf.num_train_batches)

        with tf.Session() as sess:



            sess.run(init)


            for epoch in range(self.cf.num_epochs):


                self.run_epoch(sess, batch_gen)
                self.evaluate_epoch()
                self.plot_training(sess, batch_gen)





    def run_epoch(self, sess, batch_gen):


        val_loss_running_mean = 0.
        val_dices_running_batch_mean = np.zeros(shape=(1, self.cf.num_classes))
        for _ in range(self.cf.num_val_batches):
            batch = next(batch_gen['val'])
            val_loss, val_dices, class_weights = sess.run((self.net.cost, self.net.dice_per_class, self.net.class_weights),
                                           feed_dict={self.net.x: batch['data'],
                                                      self.net.y:batch['seg'],
                                                      self.net.is_training:False,
                                                      self.net.class_weights: get_class_weights(batch['seg'])})

            print "CHECK CLASS WEIGHTS", class_weights
            val_loss_running_mean += val_loss / self.cf.num_val_batches
            val_dices_running_batch_mean[0] += val_dices / self.cf.num_val_batches

        self.metrics['val']['loss'].append(val_loss_running_mean)
        self.metrics['val']['dices'] = np.append(self.metrics['val']['dices'], val_dices_running_batch_mean, axis=0)

        train_loss_running_mean = 0.
        train_dices_running_batch_mean = np.zeros(shape=(1, self.cf.num_classes))
        for _ in range(self.cf.num_train_batches):
            batch = next(batch_gen['train'])
            train_loss, train_dices, _ = sess.run((self.net.cost, self.net.dice_per_class, self.optimizer),
                                           feed_dict={self.net.x: batch['data'],
                                                      self.net.y:batch['seg'],
                                                      self.net.is_training:True,
                                                      self.net.class_weights: get_class_weights(batch['seg'])})

            print ("LOSS", train_loss, self.epoch)
            train_loss_running_mean += train_loss / self.cf.num_train_batches
            train_dices_running_batch_mean += train_dices / self.cf.num_train_batches

        self.metrics['train']['loss'].append(train_loss_running_mean)
        self.metrics['train']['dices'] = np.append(self.metrics['train']['dices'], train_dices_running_batch_mean,
                                                   axis=0)

        self.epoch += 1



    def evaluate_epoch(self):

        val_loss = self.metrics['val']['loss'][-1]
        val_dices = self.metrics['val']['dices'][-1]

        if val_loss < self.best_metrics['loss'][0]:
            self.best_metrics['loss'][0] = val_loss
            self.best_metrics['loss'][1] = self.epoch
            # self.save_weights(spec='_{}'.format(self.fold))

        for cl in range(self.cf.num_classes):
            if val_dices[cl] > self.best_metrics['dices'][cl][0]:
                self.best_metrics['dices'][cl][0] = val_dices[cl]
                self.best_metrics['dices'][cl][1] = self.epoch


    def plot_training(self, sess, batch_gen):
        self.TrainingPlot.update_and_save(self.metrics, self.best_metrics)

        # plotting example predictions
        batch = next(batch_gen['val'])
        soft_prediction = sess.run((self.net.predicter), feed_dict={self.net.x: batch['data'], self.net.is_training:False})
        correct_prediction = np.argmax(soft_prediction, axis=3)
        outfile = self.cf.plot_dir + '/pred_examle_{}.png'.format(0) ## FOLD!
        plot_batch_prediction(batch['data'], batch['seg'], correct_prediction, self.cf.num_classes, outfile)


