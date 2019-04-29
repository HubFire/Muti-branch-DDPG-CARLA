
from environment.carla_env import Env
from image_agent.image_agent import ImageAgent
from ddpg.ActorNetwork import ActorNetwork
from ddpg.CriticNetwork import CriticNetwork
from ddpg.OU import OU 
from ddpg.replay_buffer import ReplayBuffer

import  tensorflow  as  tf 
import  random
import numpy as np  
import pickle
import random 



def train(sess,image_agent,continue_train=False):
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    GAMMA = 0.9 
    TAU = 0.001 
    INIT_LRA = 0.000001
    INIT_LRC = 0.0001 
    EPISODE_MAX_STEP = 5000
    # DECAY_RATE = 0.5 
    # DECAY_STEP = 3000000
    #TOTAL_EPISODE = 30000
    TOTAL_EPISODE = 20000
    EXPLORE = 500000
    CURRENT_STEP=0
    actor = ActorNetwork(sess,BATCH_SIZE,TAU,INIT_LRA)
    critic = CriticNetwork(sess,BATCH_SIZE,TAU,INIT_LRC)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    sess.graph.finalize()
    ou = OU()
    # if continue_train:
    #     #TODO: reload network  and  params
    #     pass
    buffer_follow = ReplayBuffer(BUFFER_SIZE)
    buffer_straight = ReplayBuffer(BUFFER_SIZE)
    buffer_left = ReplayBuffer(BUFFER_SIZE)
    buffer_right = ReplayBuffer(BUFFER_SIZE)
    buffer_dict = {0:buffer_follow,1:buffer_left,2:buffer_right,3:buffer_straight}
   
    epsilon = 1.0

    env = Env("./log","./data",image_agent)
    #env.reset()
    
    for i in range(TOTAL_EPISODE):
        try:
            ob = env.reset()
        except Exception:
            continue
        total_reward = 0
        episode_step = 0
        s_t = ob
        for j in range(EPISODE_MAX_STEP):
            if s_t is None or len(s_t)<514:
                continue
            epsilon-=1.0/ EXPLORE
            image_input = s_t[0:-2]
            speed_input = s_t[-2:-1]
            #GO_STRAIGHT = 5.0,TURN_RIGHT = 4.0,TURN_LEFT = 3.0,LANE_FOLLOW = 2.0
            direction = s_t[-1:] 
            branch_st = int(direction-2)
            if branch_st == -2:  # REACH_GOAL=0
                break
            a_t=np.zeros([1,3]) #steer throttle brake 
            noise_t = np.zeros([1,3])    
            a_t_pridect = actor.pridect_action(image_input,speed_input,branch_st)
            noise_t[0][0] = max(epsilon,0)*ou.function(a_t_pridect[0][0],0,0.6,0.3)
            noise_t[0][1] = max(epsilon,0)*ou.function(a_t_pridect[0][0],0.5,1,0.1)
            noise_t[0][2] = max(epsilon,0)*ou.function(a_t_pridect[0][0],-0.1,1,0.05)
            a_t = a_t_pridect+noise_t
            # if(CURRENT_STEP<10000) and  j<50:
            #      a_t[0][2]=0
            #      a_t[0][1]=max(0.6,a_t[0][1])
            try:
                ob,r_t,done = env.step(a_t[0])
                s_t1 = ob
                if s_t1 is None or len(s_t1)<514:
                    continue
                buffer_dict[branch_st].add(s_t,a_t[0],r_t,s_t1,done)
            except Exception:
                break

            

            # train Actor and  Critic
            branch_to_train = random.choice([0,1,2,3])
            if buffer_dict[branch_to_train].count()>128:
                train_ddpg(actor,critic,buffer_dict,BATCH_SIZE,branch_to_train)
            total_reward+=r_t
            s_t = s_t1
            CURRENT_STEP+=1
            episode_step+=1
            if (done):
                break
        
        print("buffer lenth:{},{},{},{},total reward:{},current_step:{},total_step:{}".format(buffer_dict[0].count(),
                    buffer_dict[1].count(),
                    buffer_dict[2].count(),
                    buffer_dict[3].count(),
                    total_reward,episode_step,CURRENT_STEP))
        
        if np.mod(i,2000)==0:
            saver.save(sess,'./model/ddpg_model')
            with open("./episode.txt","w") as log:
                log.write(("{},{}\n").format(i,epsilon))
            with open("./buffer.pkl","wb") as buffer_log:
                pickle.dump(buffer_dict, buffer_log)
            #TODO: save model 

def train_ddpg(actor,critic,buffer_dict,batch_size,branch):
    batch = buffer_dict[branch].getBatch(batch_size)
    #states = np.asarray([e[0][] for e in batch])  #shape = (128,514)
    states_image = np.asarray([e[0][0:-2] for e in batch])
    states_speed = np.asarray([e[0][-2:-1] for e in batch])
    actions = np.asarray([e[1] for e in batch])
    rewards = np.asarray([e[2] for e in batch])
    new_states_image = np.asarray([e[3][0:-2] for e in batch])
    new_states_speed = np.asarray([e[3][-2:-1] for e in batch])
    new_states_branch = np.asarray([e[3][-1] for e in batch])
    dones = np.asarray([e[4] for e in  batch])
    y_t = np.asarray([e[2] for e in batch])
    target_action = actor.pridect_target_action(new_states_image,new_states_speed,new_states_branch)
    target_q = critic.predict_target_q(new_states_image,new_states_speed,target_action,new_states_branch)
    for k  in range(batch_size):
        if dones[k]:
            y_t[k] = rewards[k]
        else:
            y_t[k] = rewards[k]+ 0.9*target_q[k]
    #train  critic  net
    critic.train_branch(states_image,states_speed,actions,y_t,branch)
    #train critic net
    states_branch = np.asarray([e[0][-1] for e in batch])
    action_pridect = actor.pridect_target_action(states_image,states_speed,states_branch)
    
    gradient_action = critic.run_gradient(states_image,states_speed,action_pridect,branch)
    actor.train_branch(states_image,states_speed,gradient_action,branch)
    actor.train_target()
    critic.train_target()
    
def play():
    pass
def save():
    pass    

if __name__=="__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config=tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as  sess:
        img_agent = ImageAgent(sess)
        img_agent.load_model()
        train(sess,img_agent)

             

    