import tensorflow as tf

def get_through_mlp(input, mlplayers, activ, dropout):
  mapped = input 
  for i in range(len(mlplayers)):
    mapped = tf.layers.dense(mapped, mlplayers[i], activation = activ, name = "mlplay_" + str(i+1), reuse = tf.AUTO_REUSE)
  return tf.nn.dropout(mapped, dropout)

def cosine_distance(t1, t2):
  return tf.constant(1.0, dtype = tf.float32) - tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(t1, axis = 1), tf.nn.l2_normalize(t2, axis = 1)), axis = 1)

def asym_distance(t1, t2):
    n1 = tf.norm(t1, axis = 1)
    n2 = tf.norm(t2, axis = 1)
    return tf.div(tf.subtract(n1, n2), tf.add(n1, n2))

def le_distance(t1, t2, asym_fact):
  return cosine_distance(t1, t2) + asym_fact * asym_distance(t1, t2)

def hinge_loss(true_ledists, false_ledists, margin):
  return tf.reduce_sum(tf.maximum(tf.subtract(tf.constant(margin, dtype = tf.float32), tf.subtract(false_ledists, true_ledists)), 0.0))

class ExpliLEModel(object):
  def __init__(self, embs, mlp_layers, activation = tf.nn.tanh, scope = "explile", asym_fact = 1.0, margin_le = 1.0, margin_sym = 1.0, reg_factor_hyp = 1.0, reg_factor_sym = 1.0, learning_rate = 0.0001):
    self.embeddings = embs
    self.scope = scope

    with tf.name_scope(self.scope + "__placeholders"):
    # init
      self.true_1 = tf.placeholder(tf.int32, [None,], name="t1")
      self.true_2 = tf.placeholder(tf.int32, [None,], name="t2")
      self.false_1 = tf.placeholder(tf.int32, [None,], name="f1")
      self.false_2 = tf.placeholder(tf.int32, [None,], name="f2")
      self.const_types = tf.placeholder(tf.int32, [None,], name="ct")
      self.dropout = tf.placeholder(tf.float32, name="dropout")

    with tf.name_scope(self.scope + "__model"):
      # embedding lookup
      self.embeddings = tf.get_variable("word_embeddings", initializer=embs, dtype = tf.float32, trainable = False)
      self.embs_true_1 = tf.nn.embedding_lookup(self.embeddings, self.true_1)
      self.embs_true_2 = tf.nn.embedding_lookup(self.embeddings, self.true_2)
      self.embs_false_1 = tf.nn.embedding_lookup(self.embeddings, self.false_1)
      self.embs_false_2 = tf.nn.embedding_lookup(self.embeddings, self.false_2)

      # MLPs (with or without shared parameters, depending on same_mapper)
      print("Mapping through MLP...")
      self.mapped_true_1 = get_through_mlp(self.embs_true_1, mlp_layers, activation, self.dropout)  
      self.mapped_true_2 = get_through_mlp(self.embs_true_2, mlp_layers, activation, self.dropout)
      self.mapped_false_1 = get_through_mlp(self.embs_false_1, mlp_layers, activation, self.dropout)
      self.mapped_false_2 = get_through_mlp(self.embs_false_2, mlp_layers, activation, self.dropout)  

      print("Compute distances between correct constraints and paired fake ones")
      # dists le for hyps
      self.dist_true_asym = asym_distance(self.mapped_true_1, self.mapped_true_2)
      self.dist_fake_first_asym = asym_distance(self.mapped_true_1, self.mapped_false_1)
      self.dist_fake_second_asym = asym_distance(self.mapped_false_2, self.mapped_true_2)

      # dists cosine for syns and ants 
      self.dist_true_cos = cosine_distance(self.mapped_true_1, self.mapped_true_2)
      self.dist_fake_first_cos = cosine_distance(self.mapped_true_1, self.mapped_false_1)
      self.dist_fake_second_cos = cosine_distance(self.mapped_false_2, self.mapped_true_2)
      
      # losses
      self.reg_loss = tf.reduce_sum(cosine_distance(self.embs_true_1, self.mapped_true_1)) + tf.reduce_sum(cosine_distance(self.embs_true_2, self.mapped_true_2)) + tf.reduce_sum(cosine_distance(self.embs_false_1, self.mapped_false_1)) + tf.reduce_sum(cosine_distance(self.embs_false_2, self.mapped_false_2))
      self.loss_le = tf.reduce_sum(hinge_loss(self.dist_true_asym, self.dist_fake_first_asym, margin_le)) + tf.reduce_sum(hinge_loss(self.dist_true_asym, self.dist_fake_second_asym, margin_le))

      self.loss_attract = tf.reduce_sum(hinge_loss(self.dist_true_cos, self.dist_fake_first_cos, margin_sym)) + tf.reduce_sum(hinge_loss(self.dist_true_cos, self.dist_fake_second_cos, margin_sym))
      self.loss_repel = tf.reduce_sum(hinge_loss(self.dist_fake_first_cos, self.dist_true_cos, margin_sym)) + tf.reduce_sum(hinge_loss(self.dist_fake_second_cos, self.dist_true_cos, margin_sym))
      
      self.loss_hyp = self.loss_attract + self.loss_le * asym_fact + self.reg_loss * reg_factor_hyp
      self.loss_syn = self.loss_attract + self.reg_loss * reg_factor_sym
      self.loss_ant = self.loss_repel + self.reg_loss * reg_factor_sym
      
      self.train_step_hyp = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_hyp)
      self.train_step_syn = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_syn)
      self.train_step_ant = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_ant)
  
  def replace_embs(self, embs, session):
    assign_op = self.embeddings.assign(embs)
    session.run(assign_op)