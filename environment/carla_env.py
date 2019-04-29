from carla.client import CarlaClient,make_carla_client
from carla.settings import CarlaSettings
from carla.sensor import Camera
from carla.carla_server_pb2 import Control

from carla.planner.planner import  Planner
from carla.tcp import TCPConnectionError
from carla.client import VehicleControl
import  environment.carla_config  as carla_config
from environment.carla_game import CarlaGame
#from environment.plot_position import plot_position

import  signal
import subprocess
import random
import time
import os
from PIL import Image
import numpy as np
from enum  import Enum




class action_space(object):
	def __init__(self, dim, high, low, seed):
		self.shape = (dim,)
		self.high = np.array(high)
		self.low = np.array(low)
		self.seed = seed
		assert(dim == len(high) == len(low))
		np.random.seed(self.seed)

	def sample(self):
		return np.random.uniform(self.low, self.high)

class observation_space(object):
	def __init__(self, dim, high=None, low=None, seed=None):
		self.shape = (dim,)
		self.high = high
		self.low = low
		self.seed = seed

class FinishState(Enum):
	TIME_OUT = 0
	COLLISION_VEHICLE = 1
	COLLISION_PEDESTRIAN = 2 
	COLLISION_OTHER = 3
	OFFROAD = 4

class Env(object):
	def __init__(self, log_dir,data_dir,image_agent,city="/Game/Maps/Town01"):
		self.log_dir = log_dir
		self.data_dir = data_dir
		self.carla_server_settings =None
		self.server = None
		self.server_pid = -99999
		self.game = None  #carla client 
		self.map = city
		self.host = 'localhost'
		self.port = 2000
		self.client = None
		self.is_connected = False
		self.render = None #TODO render with pygame 
		self.Image_agent = image_agent
		
		#steer,throttle,brake
		#self.action_space = action_space(3, (1.0, 1.0,1.0), (-1.0,0,0), SEED)
		#featured image,speed,steer,other lane ,offroad,
		#collision with pedestrians,vehicles,other
		#self.observation_space = observation_space(512 + 7)

		self.max_episode = 1000000
		self.time_out_step = 10000
		self.max_speed = 35
		self.speed_up_steps = 20 

		self.current_episode = 0
		self.weather = -1
		self.current_step = 0
		self.current_position = None
		self.total_reward = 0
		self.planner = None
		self.carla_setting = None
		self.number_of_vehicles = None
		self.control = None
		self.nospeed_times =0
		
		self.reward = 0
		self.observation = None
		self.done =False
	   

		self.load_config()
		self.setup_client_and_server()

	def load_config(self):
		self.vehicle_pair = carla_config.NumberOfVehicles
		self.pedestrian_pair = carla_config.NumberOfPedestrians
		self.weather_set = carla_config.set_of_weathers
		#[straight,one_curve,navigation,navigation]
		if self.map=="/Game/Maps/Town01":
			self.poses = carla_config.poses_town01()
		elif self.map=="/Game/Maps/Town02":
			self.poses = carla_config.poses_town02()
		else:
			print("Unsupported Map Name")
	
	def reset(self):
		#if not self.is_process_alive(self.server_pid):
		#self.setup_client_and_server()
		self.nospeed_times =0 
		pose_type = random.choice(self.poses)
		#pose_type =  self.poses[0]
		self.current_position = random.choice(pose_type)  #start and  end  index
		#self.current_position = (53,67)
		# self.number_of_vehicles = random.randint( self.vehicle_pair[0],self.vehicle_pair[1])
		# self.number_of_pedestrians = random.randint( self.vehicle_pair[0],self.vehicle_pair[1])
		self.number_of_vehicles = 0
		self.number_of_pedestrians = 0
		

		self.weather = random.choice(self.weather_set)
		
		settings = carla_config.make_carla_settings()
		settings.set(
			NumberOfVehicles=self.number_of_vehicles,
			NumberOfPedestrians=self.number_of_pedestrians,
			WeatherId= self.weather
		)
		self.carla_setting = settings
		self.scene = self.game.load_settings(settings)
		self.game.start_episode(self.current_position[0]) #set the start position
		#print(self.current_position)
		self.target_transform = self.scene.player_start_spots[self.current_position[1]]
		self.planner = Planner(self.scene.map_name)
		#skip the  car fall to sence frame
		for i in range(self.speed_up_steps): 
			self.control = VehicleControl()
			self.control.steer = 0
			self.control.throttle = 0.025*i
			self.control.brake = 0
			self.control.hand_brake = False
			self.control.reverse = False
			time.sleep(0.05)
			send_success = self.send_control(self.control)
			if not send_success:
				return None
			self.game.send_control(self.control)
			#measurements, sensor_data = self.game.read_data() #measurements,sensor 
			#direction =self.get_directions(measurements,self.target_transform,self.planner)
			#self.get_state(measurements,sensor_data,direction)
		measurements, sensor_data = self.game.read_data() #measurements,sensor 
		directions =self.get_directions(measurements,self.target_transform,self.planner)
		if directions is None or measurements is None:
			return None
		state,_,_=self.get_state(measurements,sensor_data,directions)
		return state 

	def get_data(self):
		measurements=None
		sensor_data=None
		try:
			measurements, sensor_data = self.game.read_data()	
		except Exception:
			return None,None
		return measurements,sensor_data
	def send_control(self,control):
		send_success = False
		try:
			self.game.send_control(control)
			send_success = True
		except Exception:
			print("Send Control error")
		return send_success

	def step(self,action):
		#take action ,update state 
		#return: observation, reward,done
		self.control = VehicleControl()
		self.control.steer = np.clip(action[0], -1, 1)
		self.control.throttle = np.clip(action[1], 0, 1)
		self.control.brake = np.abs(np.clip(action[2], 0, 1))
		self.control.hand_brake = False
		self.control.reverse = False
		send_success = self.send_control(self.control)
		if not send_success:
				return None,None,None
		#recive  new data 
		measurements, sensor_data = self.game.read_data() #measurements,sensor 
		directions =self.get_directions(measurements,self.target_transform,self.planner)
		if measurements is  None or directions is None:
			return None,None,None
		state,reward,done=self.get_state(measurements,sensor_data,directions)
		self.current_step+=1
		return state,reward,done		
		
	#comute new state,reward,and is done
	def get_state(self,measurements,sensor_data,directions):
		self.reward = 0 
		done = False 
		img_feature = self.Image_agent.compute_feature(sensor_data)  #shape = (512,)
		speed = measurements.player_measurements.forward_speed # m/s
		intersection_offroad = measurements.player_measurements.intersection_offroad
		intersection_otherlane = measurements.player_measurements.intersection_otherlane
		collision_vehicles = measurements.player_measurements.collision_vehicles
		collision_pedestrians = measurements.player_measurements.collision_pedestrians
		collision_other = measurements.player_measurements.collision_other

		# reward for steer
		if  directions == 5: #go  straight 
			if abs(self.control.steer)> 0.2: 
				self.reward-=20
			self.reward+=min(35,speed*3.6)
		elif  directions == 2: #follow  lane 
			self.reward+=min(25,speed*3.6)
		elif directions ==3: #turn  left ,steer should be negtive 
			if self.control.steer>0:
				self.reward-=15
			if speed*3.6 <=20:
				self.reward+=speed*3.6
			else:
				self.reward+= 40-speed*3.6                     
		elif directions ==4: #turn  right 
			if self.control.steer<0:
				self.reward-=15
			if speed*3.6 <=20:
				self.reward+=speed*3.6
			else:
				self.reward+= 40-speed*3.6                   
		
		# reward  for  offroad  and  collision  
		if intersection_offroad>0:
			self.reward-=100 
		if intersection_otherlane>0:
			self.reward-=100 
		elif collision_vehicles > 0:
			self.reward-=100
		elif collision_pedestrians >0:
			self.reward-=100  
		elif collision_other >0:
			self.reward-=50  
		
		# teminal  state
		if collision_pedestrians>0 or collision_vehicles>0 or collision_other >0:
			done = True
			#print("Collision~~~~~")
		if intersection_offroad>0.2 or intersection_otherlane>0.2:
			done = True
			#print("Offroad~~~~~")
		if speed*3.6 <=1.0:
			self.nospeed_times+=1
			if self.nospeed_times>100:
				done=True
			self.reward-=1
		else:
			self.nospeed_times=0
		# compute  state  512+2
		speed = min(1,speed/10.0)
		
		return  np.concatenate((img_feature, (speed,directions))),self.reward,done 
	def _open_server(self):
		with open(self.log_dir, "wb") as out:
			cmd = [os.path.join(os.environ.get('CARLA_ROOT'), 'CarlaUE4.sh'),
					self.map,
					 "-carla-server", "-fps=10", "-world-port={}".format(
						self.port),
					"-windowed -ResX={} -ResY={}".format(
						carla_config.WINDOW_WIDTH,carla_config.WINDOW_HEIGHT)
					]
			if self.carla_server_settings:
				cmd.append("-carla-settings={}".format(self.carla_server_settings))
			p = subprocess.Popen(cmd, stdout=out, stderr=out)
			time.sleep(20)
		return p
	# This  is not work 
	def _close_server(self):
		no_of_attempts = 0
		try:
			while self.is_process_alive(self.server_pid):
				print("Trying to close Carla server with pid %d" % self.server_pid)
				if no_of_attempts < 5:
					self.server.terminate()
				elif no_of_attempts < 10:
					self.server.kill()
				elif no_of_attempts < 15:
					os.kill(self.server_pid, signal.SIGTERM)
				else:
					os.kill(self.server_pid, signal.SIGKILL)
				time.sleep(10)
				no_of_attempts += 1
		except Exception as  e:
			print(e)
	def close_server(self):
		sudopass="huyi123"
		command = "echo {0} |sudo -S kill -9 {1}".format(sudopass,self.server_pid)
		os.system(command)
	def is_process_alive(self,pid):
		## Source: https://stackoverflow.com/questions/568271/how-to-check-if-there-exists-a-process-with-a-given-pid-in-python
		try:
			os.kill(pid, 0)
		except OSError:
			return False
		return True
	def setup_client_and_server(self):
		# self.server = self._open_server()
		# self.server_pid = self.server.pid
		self.game = CarlaClient(self.host, self.port, timeout=99999999) #carla  client 
		self.game.connect(connection_attempts=100)
	def get_directions(self,measurements, target_transform, planner):
		""" Function to get the high level commands and the waypoints.
			The waypoints correspond to the local planning, the near path the car has to follow.
		"""

		# Get the current position from the measurements
		current_point = measurements.player_measurements.transform
		try:
			directions = planner.get_next_command(
				(current_point.location.x,
					current_point.location.y, 0.22),
				(current_point.orientation.x,
					current_point.orientation.y,
					current_point.orientation.z),
				(target_transform.location.x, target_transform.location.y, 0.22),
				(target_transform.orientation.x, target_transform.orientation.y,
					target_transform.orientation.z)
			)
		except Exception:
			print("Route plan error ")
			directions = None
		return directions

# if __name__=="__main__":
#     with tf.Session() as  sess:
#         env = Env("./log","./data",sess)
#         #env.setup_client_and_server()
#         env.reset()
	
   
