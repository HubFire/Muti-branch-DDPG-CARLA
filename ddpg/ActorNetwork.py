import tensorflow as tf
import numpy as np
def weight_xavi_init(shape,name):
    return tf.get_variable(shape=shape,name=name,initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape,name):
    return tf.get_variable(shape=shape,name=name,initializer=tf.constant_initializer(0.1))

def fc(x, output_size,layer_name):
        filters_in = x.get_shape()[-1]
        shape = [filters_in, output_size]
        weights = weight_xavi_init(shape,layer_name+"_w")
        bias = bias_variable([output_size],layer_name+"_b")
        return tf.nn.relu(tf.nn.xw_plus_b(x, weights,bias))

class ActorNetwork(object):
    def __init__(self,sess,batch_size,tau,learning_rate):
        self.sess = sess       
        self.batch_size = batch_size 
        self.tau = tau 
        self.learning_rate = learning_rate
        self.image_input,self.speed_input,self.branches = \
            self.create_actor_network(scope="Actor")
        self.target_image_input,self.target_speed_input,self.target_branches = \
            self.create_actor_network(scope="TargetActor")
        
        self.action_gradient = tf.placeholder(tf.float32,[None, 3])   
        self.branch_input =  tf.placeholder(tf.float32,[None, 1])
        self.branch_optimizes = [self.get_branch_optimize(i) for i in range(4)]
        self.target_optimize = self.get_target_optimize()
        #self.get_select_action_op = self.get_select_op()

        
    def create_actor_network(self,scope):
        branches = [] #4 branches:follow,straight,turnLeft,turnRight 
        with tf.variable_scope(scope):
            with tf.variable_scope("Share"):
                image_input = tf.placeholder(dtype=tf.float32,shape=(None,512),name="Actor_ImageInput")
                speed_input = tf.placeholder(dtype=tf.float32,shape=(None,1),name="Actor_SpeedInput")
                speed_fc1 = fc(speed_input,128,"speed_layer_1")
                speed_fc2 = fc(speed_fc1,128,"speed_layer_2")
                x_fc = tf.concat([image_input, speed_fc2], 1)
                x_fc = fc(x_fc,512,"concat_fc")
            for i in range(4):
                scope_name = "branch_{}".format(i)
                with tf.name_scope(scope_name):
                    branch_output = fc(x_fc,256,scope_name+"_layer1")
                    branch_output = fc(branch_output,256,scope_name+"_layer2")
                    branch_output = fc(branch_output,3,scope_name+"_out")
                branches.append(branch_output)
        return image_input,speed_input,branches
    def get_weights(self,scope):
        all_weights =[]
        params_fc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
        all_weights.extend(params_fc)
        return all_weights
    # Share weights + one branch weights
    def get_weights_branch(self,branch_num):
        branch_weights = []
        params_share = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Actor/Share")
        scope_name = scope_name = "Actor/branch_{}".format(branch_num)
        params_branch = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope_name)
        branch_weights.extend(params_share)
        branch_weights.extend(params_branch)
        return branch_weights
    def get_branch_optimize(self,branch_num):
        branch_out = self.branches[branch_num]
        branch_params =self.get_weights_branch(branch_num)
        params_grad = tf.gradients(branch_out, branch_params, self.action_gradient)
        grads = zip(params_grad, branch_params)
        # -self.learning_rate  for ascent policy 
        branch_optimize = tf.train.AdamOptimizer(-self.learning_rate).apply_gradients(grads)
        return branch_optimize
    def get_target_optimize(self):
        params = self.get_weights(scope="Actor") 
        target_params = self.get_weights(scope="TargetActor")
        target_optimize=[tf.assign(t,(1-self.tau)*t+self.tau*e) for t,e in zip(target_params,params)]
        return target_optimize

    def train_branch(self,image_input,speed_input,action_grads,branch_num):
        self.sess.run(self.branch_optimizes[branch_num],feed_dict={
            self.image_input:image_input,
            self.speed_input:speed_input,
            self.action_gradient:action_grads
        })
    def train_target(self):
        self.sess.run(self.target_optimize)
    def pridect_action(self,image_input,speed_input,branch_num):
        action = self.sess.run(self.branches[branch_num],feed_dict={
            self.image_input:np.reshape(image_input,(-1,512)),
            self.speed_input:np.reshape(speed_input,(-1,1))
        })
        return action

    def pridect_target_action(self,image_input,speed_input,branch_input):
      
        action_branchs = self.sess.run(self.target_branches,feed_dict={
            self.target_image_input:np.reshape(image_input,(self.batch_size,512)),
            self.target_speed_input:np.reshape(speed_input,(self.batch_size,1))
        })  # action_branchs is list 
        # print(len(action_branchs),)
        index = branch_input.astype(np.int16) - 2
        #index =np.cast(branch_input, tf.int32) - 2
        #branch_idx = np.stack([np.arange(self.batch_size), index], axis=1)  #shape = (?,2)
        action_branchs = np.stack(action_branchs[:4], axis=1)
        selected_branch = []
        for i in range(self.batch_size):
            selected_branch.append(action_branchs[i][index[index[i]]])
        #selected_branch = tf.gather_nd(params=action_branchs, indices=branch_idx)  
        return selected_branch
