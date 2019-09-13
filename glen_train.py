import util
import model
import numpy as np
import tensorflow as tf
import pickle
import sys
import random
import logger
from scipy import stats
import os
from sys import stdin
import itertools
import argparse
import config as c

curr_dir = os.path.dirname(__file__)

# hyperparameter configurations
drop_vals = [0.5, 1.0]
asym_fact_vals = [0.5, 1.0, 2.0]
hinge_margin_vals = [0.5, 1.0, 2.0]
margin_sym_vals = [0.5, 1.0, 2.0]
reg_factor_hyp_vals = [0.5, 1.0]
reg_factor_sym_vals = [0.5, 1.0]

configs = list(itertools.product(drop_vals, asym_fact_vals, hinge_margin_vals, margin_sym_vals, reg_factor_hyp_vals, reg_factor_sym_vals))

# best hyperparameter configuration is at index [87]
print("Hyperparameter configuration: " + str(configs[87]))
config_ind = 87
drp, af, hm, sm, rfh, rfs = configs[config_ind]

PARAMETERS = { "model_name": c.model_name,
               "log_path": os.path.join(curr_dir, c.model_dir, "output.log"),
               "ckpt_path" : os.path.join(curr_dir, c.model_dir),
               "emb_size": 300, 
               "mlp_lay" : [300]*5,
               "dropout": drp,
               "asym_fact": af,
               "hinge_margin": hm,
               "margin_sym": sm, 
               "reg_factor_hyp": rfh, 
               "reg_factor_sym": rfs, 
               "learning_rate": 0.0001,
               "num_dev": 2000,
               "batch_size": 50,
               "eval_steps": 1000,
               "num_evals_not_better_end": 10}       

print("Loading embeddings...")
vectors = np.load(os.path.join(curr_dir, c.vectors_path))
vocab = pickle.load(open(os.path.join(curr_dir, c.vocab_path),"rb")) 
reshaped_vectors = vectors

logger = logger.Logger(PARAMETERS["log_path"])

# dataset loading
train = pickle.load(open(os.path.join(curr_dir, c.train_path),"rb")) 
dev = pickle.load(open(os.path.join(curr_dir, c.dev_path),"rb")) 

train_hyps = [x for x in train if x[2] == "h"]
train_syns = [x for x in train if x[2] == "s"]
train_ants = [x for x in train if x[2] == "a"]
dev_hyps = [x for x in dev if x[2] == "h"]
dev_syns = [x for x in dev if x[2] == "s"]
dev_ants = [x for x in dev if x[2] == "a"]

class modelExecutor:
  def __init__(self):
    # model initialization
    self.model = model.ExpliLEModel(vectors, PARAMETERS["mlp_lay"], activation = tf.nn.tanh, scope = "exp_le", asym_fact = PARAMETERS["asym_fact"], margin_le = PARAMETERS["hinge_margin"], reg_factor_hyp = PARAMETERS["reg_factor_hyp"], reg_factor_sym = PARAMETERS["reg_factor_sym"], learning_rate = PARAMETERS["learning_rate"], margin_sym = PARAMETERS["margin_sym"])
    self.asym_fact = PARAMETERS["asym_fact"]
    self.batch_size = PARAMETERS["batch_size"]
    self.keep_rate = PARAMETERS["dropout"]
    self.eval_steps = PARAMETERS["eval_steps"]

    logger.Log("Initializing variables")
    self.init = tf.global_variables_initializer()
    self.sess = None
    self.saver = tf.train.Saver()

  def get_minibatch(self, pairs):
      w1s = []; w2s = []; f1s = []; f2s = []

      dist_words = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
      dist_vecs = np.array([reshaped_vectors[vocab[w]] for w in dist_words])
      norms = np.linalg.norm(dist_vecs, axis = 1)
      dv_norm = dist_vecs / norms[:, np.newaxis]       
       
      cos_dist = 1 - np.dot(dv_norm, np.transpose(dv_norm))
      n1s = np.repeat([norms], len(norms), axis = 1).reshape((len(norms), len(norms)))
      n2s = np.repeat([norms], len(norms), axis = 0)
      asym_dist = (n1s - n2s) / (n1s + n2s)
      all_dists = cos_dist + self.asym_fact * asym_dist

      for p in pairs: 
        ind_w1 = vocab[p[0]]
        ind_w2 = vocab[p[1]] 
        ind_fake_w1 = None
        ind_fake_w2 = None

        dists_to_sort = all_dists if p[2] == "h" else cos_dist 

        w1dists = np.argsort(dists_to_sort[dist_words.index(p[0])]) if p[2] != "a" else list(reversed(np.argsort(dists_to_sort[dist_words.index(p[0])])))
        for i in range(len(w1dists)):
          w_cand = dist_words[w1dists[i]]
          if w_cand != p[0] and w_cand != p[1]:
            ind_fake_w1 = vocab[w_cand]
            break 

        w2dists = np.argsort(dists_to_sort[:, dist_words.index(p[1])]) if p[2] != "a" else list(reversed(np.argsort(dists_to_sort[:, dist_words.index(p[1])])))
        for i in range(len(w2dists)):
          w_cand = dist_words[w2dists[i]]
          if w_cand != p[0] and w_cand != p[1]:
            ind_fake_w2 = vocab[w_cand] 
            break
        
        if ind_fake_w1 is None or ind_fake_w2 is None:
          print("Indices of words for negative pairs not assigned!")
          continue 
        w1s.append(ind_w1)
        w2s.append(ind_w2)
        f1s.append(ind_fake_w1)  
        f2s.append(ind_fake_w2)
      return w1s, w2s, f1s, f2s

  def train_model(self):
    self.step = 0
    self.epoch = 0
    self.best_dev = 100000000
    self.best_hl = 0
    self.best_mtrain = 100000000
    self.last_train = [1000000, 1000000, 1000000, 1000000, 1000000]
    self.best_step = 0

    self.sess = tf.Session()
    self.sess.run(self.init)

    # Restore most recent checkpoint if it exists. 
    ckpt_file = os.path.join(PARAMETERS["ckpt_path"], PARAMETERS["model_name"]) + ".ckpt"
    if os.path.isfile(ckpt_file + ".meta"):
      if os.path.isfile(ckpt_file + "_best.meta"):
        self.saver.restore(self.sess, (ckpt_file + "_best"))
        self.best_dev = self.eval()
        logger.Log("Restored best dev f1: %f\n" % (self.best_dev))
        self.saver.restore(self.sess, ckpt_file) 
        logger.Log("Model restored from file: %s" % ckpt_file)

    ### Training cycle
    logger.Log("Training...")
    reshaped_vectors = vectors
    while True:
      epoch_loss = 0.0

      random.shuffle(train_hyps) 
      random.shuffle(train_syns)
      random.shuffle(train_ants)

      num_batch_hyps = int(len(train_hyps) / self.batch_size) if len(train_hyps) % self.batch_size == 0 else (int(len(train_hyps) / self.batch_size) + 1)
      batches_hyps = [train_hyps[i * self.batch_size : (i+1) * self.batch_size] for i in range(num_batch_hyps)] 
      print(len(batches_hyps))
      
      num_batch_syns = int(len(train_syns) / self.batch_size) if len(train_syns) % self.batch_size == 0 else (int(len(train_syns) / self.batch_size) + 1)
      batches_syns = [train_syns[i * self.batch_size : (i+1) * self.batch_size] for i in range(num_batch_syns)]
      print(len(batches_syns))

      num_batch_ants = int(len(train_ants) / self.batch_size) if len(train_ants) % self.batch_size == 0 else (int(len(train_ants) / self.batch_size) + 1)
      batches_ants = [train_ants[i * self.batch_size : (i+1) * self.batch_size] for i in range(num_batch_ants)]
      print(len(batches_ants))

      batches = batches_hyps + batches_syns + batches_ants + batches_ants
      random.shuffle(batches)

      # Loop over all batches in epoch
      for batch in batches:
        w1s, w2s, f1s, f2s = self.get_minibatch(batch)        
        feed_dict = {self.model.true_1: w1s,
                     self.model.true_2: w2s,
                     self.model.false_1: f1s,
                     self.model.false_2: f2s, 
                     self.model.dropout: self.keep_rate }
         
        if batch[0][2] == "h":
          cl1, catt, creg = self.sess.run([self.model.loss_le, self.model.loss_attract, self.model.reg_loss], feed_dict)
          _, c = self.sess.run([self.model.train_step_hyp, self.model.loss_hyp], feed_dict)
          
        elif batch[0][2] == "a":
          crep, creg = self.sess.run([self.model.loss_repel, self.model.reg_loss], feed_dict)
          _, c = self.sess.run([self.model.train_step_ant, self.model.loss_ant], feed_dict)  
          
        elif batch[0][2] == "s":
          catt, creg = self.sess.run([self.model.loss_attract, self.model.reg_loss], feed_dict)
          _, c = self.sess.run([self.model.train_step_syn, self.model.loss_syn], feed_dict)    
        else:
          raise ValueError("Unknown batch type!")  

        print("Batch " + str(self.step) + ", " + batch[0][2] + ": " + str(c)) 
        epoch_loss += c

        if self.step % self.eval_steps == 0:
          dev_perf = self.eval()
          logger.Log("Step: %i\t Dev perf: %f" %(self.step, dev_perf))
          self.saver.save(self.sess, ckpt_file)

          print("Saving model...")  
          if dev_perf < 0.999 * self.best_dev:
            print("New best model found...") 
            print("New best dev loss: " + str(dev_perf)) 
            self.saver.save(self.sess, ckpt_file + "_best")
            self.best_dev = dev_perf
            self.best_step = self.step
            logger.Log("Checkpointing with new best matched-dev loss: %f" %(self.best_dev))
          elif (self.step > self.best_step + (self.eval_steps * PARAMETERS["num_evals_not_better_end"] + 10)):
            print("Exit condition (early stopping) met.")  
            logger.Log("Best matched-dev loss: %s" % (self.best_dev))
            print("Training successfully finished.")  
            return

        self.step += 1
                                  
      # Display some statistics about the epoch
      logger.Log("Epoch: %i\t Avg. Cost: %f" %(self.epoch+1, epoch_loss / len(train)))
      self.epoch += 1 
      epoch_loss = 0.0 

  def eval(self):
    total_dev_loss = 0.0
    dev_num_batch_hyps = int(len(dev_hyps) / self.batch_size) if len(dev_hyps) % self.batch_size == 0 else (int(len(dev_hyps) / self.batch_size) + 1)
    dev_batches_hyps = [dev_hyps[i * self.batch_size : (i+1) * self.batch_size] for i in range(dev_num_batch_hyps)] 
    
    dev_num_batch_syns = int(len(dev_syns) / self.batch_size) if len(dev_syns) % self.batch_size == 0 else (int(len(dev_syns) / self.batch_size) + 1)
    dev_batches_syns = [dev_syns[i * self.batch_size : (i+1) * self.batch_size] for i in range(dev_num_batch_syns)]

    dev_num_batch_ants = int(len(dev_ants) / self.batch_size) if len(dev_ants) % self.batch_size == 0 else (int(len(dev_ants) / self.batch_size) + 1)
    dev_batches_ants = [dev_ants[i * self.batch_size : (i+1) * self.batch_size] for i in range(dev_num_batch_ants)]

    dev_batches = dev_batches_hyps + dev_batches_syns + dev_batches_ants

    # Loop over all batches in dev set
    total_dev_loss = 0.0
    for db in dev_batches:
      w1s, w2s, f1s, f2s = self.get_minibatch(db)
      feed_dict = {self.model.true_1: w1s,
                   self.model.true_2: w2s,
                   self.model.false_1: f1s,
                   self.model.false_2: f2s, 
                   self.model.dropout: self.keep_rate }

      c = 0.0
      if db[0][2] == "h":
        c = self.sess.run(self.model.loss_hyp, feed_dict) 
      elif db[0][2] == "a":
        c = self.sess.run(self.model.loss_ant, feed_dict) 
      elif db[0][2] == "s":
        c = self.sess.run(self.model.loss_syn, feed_dict) 
      total_dev_loss += c
    return total_dev_loss / len(dev)

me = modelExecutor()
me.train_model()