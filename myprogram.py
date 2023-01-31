import glob
import os
import sys
import random
import time
import numpy as np
import cv2 #to use rgb easier
import math
from collections import deque

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SHOW_PREVIEW = False # to see better the agents, but this need a great CPU
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000 #5,000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDCTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.8
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0 
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__ (self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library
        self.model_3 = self.blueprint_library.filter("model3")[0] # the model of the car
    
    def reset(self):
        self.colission_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb.set_attribute("image_size_x", f"110")

        transform = carla.Transform(carla.Location(x = 2.5, z = 0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.colission")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.colission_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def colission_data(self, event):
        self.colission_hist.append(event)


    def process_img(self, image):
        i = np.array(image.raw_data) #numpy array of the image
        #print(dir(image)) #print image
        i2 = i.reshape((self.im_height, self.im_width, 4 )) #reshape the array of the image to coordenates ("4" is because i'm using rgb)
        i3 = i2[:, :, :3] # is to use all of the height and the width to rgb (because the fourth value is alpha and i don't know how to use alpha values)
        if self.SHOW_CAM:
            cv2.imshow("", i3) # representation of what the camera sees
            cv2.waitKey(1)  # representation of what the camera sees
        self.show_camera = i3
    
    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = -1*self.STEER_AMT)) # turn left
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = 0)) # go straight
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = 1*self.STEER_AMT)) # turn right

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50: # This is to prevent the car from going around in circles
            done = False
            reward = -1;
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE  < time.time(): #time out of episodes
            done = True
    
        return self.front_camera, reward, done, None
    
class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxLen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")