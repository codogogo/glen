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

simlex = [(l.split("\t")[0].lower(), l.split("\t")[1].lower(), float(l.split("\t")[3])) for l in util.load_lines("/work/gglavas/data/evaluation/simlex999/SimLex-999.txt")]
wsim = [(l.split("\t")[0], l.split("\t")[1], float(l.split("\t")[2])) for l in util.load_lines("/work/gglavas/data/evaluation/wsim353/wsim_sim.txt")]

curr_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description="Predicting graded lexical entailment using a previously trained GLEN model.")
parser.add_argument("pairs_path", type=str, help="Path to the file containing word pairs for which to predict the graded LE (format: each line 'word1[TAB]word2')")
parser.add_argument("preds_path", type=str, help="Path to the file where the predictions are to be saved.")
args = parser.parse_args()

# hyperparameter configurations
drop_vals = [0.5, 1.0]
asym_fact_vals = [0.5, 1.0, 2.0]
hinge_margin_vals = [0.5, 1.0, 2.0]
margin_sym_vals = [0.5, 1.0, 2.0]
reg_factor_hyp_vals = [0.5, 1.0]
reg_factor_sym_vals = [0.5, 1.0]

configs = list(itertools.product(drop_vals, asym_fact_vals, hinge_margin_vals, margin_sym_vals, reg_factor_hyp_vals, reg_factor_sym_vals))
print(configs[0])
print(len(configs))

config_ind = 87
drp, af, hm, sm, rfh, rfs = configs[config_ind]
print("Configuration: ")
print(drp, af, hm, sm, rfh, rfs)
print()
print()

PARAMETERS = { "model_name": c.model_name,
               "log_path": os.path.join(curr_dir, c.model_dir, "output_" + str(config_ind) + ".log"),
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

print("Loading prediction pairs...")
#testpairs = util.load_csv_lines(args.pairs_path, delimiter = "\t")
#testpairs = [(x[0], x[1]) for x in testpairs]
testpairs = simlex

vectors = np.load(os.path.join(curr_dir, c.vectors_path))
vocab = pickle.load(open(os.path.join(curr_dir, c.vocab_path),"rb")) 

w1s = []
w2s = []
valid_pairs = [] 

for pair in testpairs:
  if (pair[0].strip() not in vocab and pair[0].strip().lower() not in vocab) or (pair[1].strip() not in vocab and pair[1].strip().lower() not in vocab):
    print("At least one of the words in the pair not found in the distributional vocabulary: " + str(pair))
  else: 
    w1s.append(vocab[pair[0].strip()] if pair[0].strip() in vocab else vocab[pair[0].strip().lower()])  
    w2s.append(vocab[pair[1].strip()] if pair[1].strip() in vocab else vocab[pair[1].strip().lower()])
    valid_pairs.append(pair)
    
class modelExecutor:
  def __init__(self):
    # model initialization
    self.model = model.ExpliLEModel(vectors, PARAMETERS["mlp_lay"], activation = tf.nn.tanh, scope = "exp_le", asym_fact = PARAMETERS["asym_fact"], margin_le = PARAMETERS["hinge_margin"], reg_factor_hyp = PARAMETERS["reg_factor_hyp"], reg_factor_sym = PARAMETERS["reg_factor_sym"], learning_rate = PARAMETERS["learning_rate"], margin_sym = PARAMETERS["margin_sym"])   
    self.init = tf.global_variables_initializer()
    self.sess = None
    self.saver = tf.train.Saver()

  def restore_best(self):
    self.sess = tf.Session()
    self.sess.run(self.init)

    # Restore most recent checkpoint if it exists. 
    print("Loading the best model...")
    ckpt_file = os.path.join(PARAMETERS["ckpt_path"], PARAMETERS["model_name"]) + ".ckpt"
    if os.path.isfile(ckpt_file + ".meta"):
      if os.path.isfile(ckpt_file + "_best.meta"):
        self.saver.restore(self.sess, (ckpt_file + "_best"))
    else:
      print("Model not found. Check the model path parameters in the config file (model_dir and model_name).")
      exit()

  def evaluate(self):  
    feed_dict_1 = {self.model.true_1: w1s,
                 self.model.dropout: 1.0 }
    feed_dict_2 = {self.model.true_1: w2s,
                 self.model.dropout: 1.0 }

    vecs_first = self.sess.run(self.model.mapped_true_1, feed_dict_1)
    vecs_second = self.sess.run(self.model.mapped_true_1, feed_dict_2)
    mapped_vec_pairs =  list(zip(vecs_first, vecs_second))

    pred_scores = []
    for vec1, vec2 in mapped_vec_pairs:
    #for i in range(len(w1s)):
      #vec1 = vectors[w1s[i]]
      #vec2 = vectors[w2s[i]]
      #print(vec1)
      #print(vec2)
      #stdin.readline()
      cos_dist = (1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
      cos_sim = 1 - cos_dist
      asym_dist = (np.linalg.norm(vec1) - np.linalg.norm(vec2)) / (np.linalg.norm(vec1) + np.linalg.norm(vec2))
      score = cos_dist + 1.0 * asym_dist
      #pred_scores.append(score)
      print(cos_sim)
      pred_scores.append(cos_dist) 
    return pred_scores

me = modelExecutor()
me.restore_best()
preds = me.evaluate()

if len(valid_pairs) != len(preds):
  raise ValueError("Unexpected number of predictions!")

print("SIMLEX")
correl = stats.spearmanr(preds, [x[2] for x in valid_pairs])
print(correl)

out_lines = [valid_pairs[i][0] + "\t" + valid_pairs[i][1] + "\t" + str(preds[i]) for i in range(len(valid_pairs))]
util.write_lines(args.preds_path, out_lines)
print("Predictions successfully written to the output file.")