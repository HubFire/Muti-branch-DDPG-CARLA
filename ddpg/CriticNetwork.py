import  tensorflow  as  tf
import numpy as  np 
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

class CriticNetwork(object):
    def __init__(self,sess,batch_size,tau,learning_rate):
        self.sess = sess       
        self.batch_size = batch_size 
        self.tau = tau 
        self.learning_rate = learning_rate
        self.image_input,self.speed_input,self.action_input,self.branches = \
            self.create_critic_network(scope="Critic")
        self.target_image_input,self.target_speed_input,self.target_action,self.target_branches = \
            self.create_critic_network(scope="TargetCritic")
        self.target_q = tf.placeholder(tf.float32,[None, 1])   
        self.action_gradients = [self.get_gradient(i) for i in range(4)]
        self.target_optimize = self.get_target_optimize()
        self.branch_optimizes = [self.get_branch_optimize(i) for i in range(4)]


    def create_critic_network(self,scope):
        branches = [] #4 branches:follow,straight,turnLeft,turnRight
        with tf.variable_scope(scope):
            with tf.variable_scope("Share"):
                image_input = tf.placeholder(dtype=tf.float32,shape=(None,512),name="Critic_ImageInput")
                speed_input = tf.placeholder(dtype=tf.float32,shape=(None,1),name="Critic_SpeedInput")
                action_input = tf.placeholder(dtype=tf.float32,shape=(None,3),name="Critic_ActionInput")
                speed_fc1 = fc(speed_input,128,"speed_layer_1")
                speed_fc2 = fc(speed_fc1,128,"speed_layer_2")

                action_fc1 = fc(action_input,128,"action_layer_1")
                action_fc2 = fc(action_fc1,128,"action_layer_2")

                x_fc = tf.concat([image_input, speed_fc2,action_fc2], 1)
                x_fc = fc(x_fc,512,"concat_fc")
            for i in range(4):
                scope_name = "branch_{}".format(i)
                with tf.name_scope(scope_name):
                    branch_output = fc(x_fc,256,scope_name+"_layer1")
                    branch_output = fc(branch_output,256,scope_name+"_layer2")
                    branch_output = fc(branch_output,1,scope_name+"_out")
                branches.append(branch_output)
        return image_input,speed_input,action_input,branches
    def get_gradient(self,branch_num):
        return tf.gradients(self.branches[branch_num],self.action_input)
    def run_gradient(self,image,speed,action,branch_num):
        return self.sess.run(self.action_gradients[branch_num],feed_dict={
            self.image_input:image,
            self.speed_input:speed,
            self.action_input:action
        })[0]
        
    def get_target_optimize(self):
        params = self.get_weights(scope="Critic") 
        target_params = self.get_weights(scope="TargetCritic")
        target_optimize=[tf.assign(t,(1-self.tau)*t+self.tau*e) for t,e in zip(target_params,params)]
        return target_optimize
    def train_target(self):
        self.sess.run(self.target_optimize)
    def get_weights(self,scope):
        all_weights =[]
        params_fc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
        all_weights.extend(params_fc)
        return all_weights
    def get_weights_branch(self,branch_num):
        branch_weights = []
        params_share = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Critic/Share")
        scope_name = scope_name = "Critic/branch_{}".format(branch_num)
        params_branch = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope_name)
        branch_weights.extend(params_share)
        branch_weights.extend(params_branch)
        return branch_weights

    def get_branch_optimize(self,branch_num):
        branch_out = self.branches[branch_num]
        branch_params =self.get_weights_branch(branch_num)
        loss = tf.reduce_mean(tf.squared_difference(self.target_q,branch_out))
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss,var_list=branch_params)
    def train_branch(self,image_input,speed_input,action_input,target_q,branch_num):
        self.sess.run(self.branch_optimizes[branch_num],feed_dict={
            self.image_input:image_input,
            self.speed_input:speed_input,
            self.action_input:action_input,
            self.target_q:np.reshape(target_q,(-1,1))
        })

    def predict_q(self,image_input,speed_input,action_input,branch_num):
        return self.sess.run(self.branches[branch_num],feed_dict={
            self.image_input:image_input,
            self.speed_input:speed_input,
            self.action_input:action_input
        })
    
    def predict_target_q(self,image_input,speed_input,action_input,branch_input):
        q_actions =  self.sess.run(self.target_branches,feed_dict={
            self.target_image_input:image_input,
            self.target_speed_input:speed_input,
            self.target_action:action_input
        })

        index = branch_input.astype(np.int16) - 2
        #branch_idx = tf.stack([tf.range(self.batch_size), index], axis=1)  #shape = (?,2)
        q_branchs = np.stack(q_actions[:4], axis=1)

        selected_branch = []
        for i in range(self.batch_size):
            selected_branch.append(q_branchs[i][index[index[i]]])
        #selected_branch = tf.gather_nd(params=q_branchs, indices=branch_idx)  
        return selected_branch
