import time
import logging
import math

try:
    import pygame
    from pygame.locals import K_r
    from pygame.locals import K_p
    from pygame.locals import K_ESCAPE
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla.planner.planner import sldist
from carla.planner.map import CarlaMap
from carla import image_converter


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


def vector_to_degrees(vector):

    return math.atan2(-vector[1], vector[0])


class CarlaGame(object):
    """
        Class to plot a game screen and control a simple situation were the player has
        to reach some objectives.
        Based on the manual_control.py script from carla 0.8.4
    """

    def __init__(self, autopilot, display_map, window_width, window_height, mini_window_width, mini_window_height):
        self._render_iter = 0
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._display_map = display_map
        self._map_name = None
        self._map = None
        self._map_shape = None
        self._map_view = None
        self._goal_position = None
        self._render_mode = False
        self._window_width = window_width
        self._window_height = window_height
        self._mini_window_width = mini_window_width
        self._mini_window_height = mini_window_height

    def initialize_game(self, map_name, render_mode=True):
        """
            Initialize the windows
        Args:
            map_name: The map that is going to be used (If it was up to display)

        Returns:
            None

        """
        self._render_mode = render_mode
        self._map_name = map_name
        self._map = CarlaMap(map_name, 0.1643, 50.0)
        self._map_shape = self._map.map_image.shape
        self._map_view = self._map.get_map(self._window_height)

        if self._render_mode:
            if self._display_map:

                extra_width = int(
                    (self._window_height / float(self._map_shape[0])) * self._map_shape[1])
                self._display = pygame.display.set_mode(
                    (self._window_width + extra_width, self._window_height),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
            else:
                self._display = pygame.display.set_mode(
                    (self._window_width, self._window_height),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)

            logging.debug('pygame started')
    """
        ****************************
        EPISODE CONTROLLING FUNCTIONS
        ****************************
    """
    def start_timer(self):
        self._timer = Timer()

    def set_objective(self, goal_position):
        """
            Set the player objective, the goal position.
            This will be rendered in the map in the future
            Only sets if map is enabled.
        Args:
            goal_position:

        Returns:

        """

        goal_position = self._map.convert_to_pixel([
            goal_position.location.x, goal_position.location.y, goal_position.location.z])

        self._goal_position = goal_position

    def is_reset(self, player_position):
        """
            Check if player reach the goal or if reset button is pressed
        Args:
            player_position: The player position

        Returns:

        """

        player_position = self._map.convert_to_pixel([
            player_position.x, player_position.y, player_position.z])

        if sldist(player_position, self._goal_position) < 7.0:
            return True

        if not self._render_mode:
            return False

        keys = pygame.key.get_pressed()
        return keys[K_r]

    def is_running(self):
        """
            If esc is not pressed the game is running
        Returns:
            if esc was pressed
        """
        if not self._render_mode:
            return True
        keys = pygame.key.get_pressed()
        return not keys[K_ESCAPE]

    def is_autopilot_enabled(self):
        keys = pygame.key.get_pressed()
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot

        return self._enable_autopilot

    """
        *******************
        RENDERING FUNCTIONS
        *******************
    """

    def render(self, camera_to_render, objects_to_render):

        """
        Main rendering function. Render a main camera and a map containing several objects
        Args:
            camera_to_render: The sensor data, images you want to render on the window
            objects_to_render: A dictionary Several objects to be rendered
                player_position: The current player position
                waypoints: The waypoints , next positions. If you want to plot then.
                agents_positions: All agent positions ( vehicles)

        Returns:
            None
        """

        if camera_to_render is not None:
            array = image_converter.to_rgb_array(camera_to_render)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        # only if the map view setting is set we actually plot all the positions and waypoints
        if self._map_view is not None:

            player_position = self._map.convert_to_pixel([
                objects_to_render['player_transform'].location.x,
                objects_to_render['player_transform'].location.y,
                objects_to_render['player_transform'].location.z])
            player_orientation =  objects_to_render['player_transform'].orientation

            array = self._map_view
            array = array[:, :, :3]

            new_window_width = \
                (float(self._window_height) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            # Draw other two fovs
            if objects_to_render['fov_list'] is not None:
                fov_1 = objects_to_render['fov_list'][0]
                fov_2 = objects_to_render['fov_list'][1]

                self._draw_fov(surface, player_position,
                               vector_to_degrees([player_orientation.x, player_orientation.y]),
                               radius=fov_1[0]/(0.1643*2), angle=fov_1[1], color=[255, 128, 0, 255])
                self._draw_fov(surface, player_position,
                               vector_to_degrees([player_orientation.x, player_orientation.y]),
                               radius=fov_2[0]/(0.1643*2), angle=fov_2[1], color=[128, 64, 0, 255])

            if objects_to_render['waypoints'] is not None:
                self._draw_waypoints(surface, objects_to_render['waypoints'])
            if objects_to_render['route'] is not None:
                self._draw_route(surface, objects_to_render['route'])
            self._draw_goal_position(surface)

            # Draw the player positions
            w_pos = int(
                player_position[0] * (float(self._window_height) / float(self._map_shape[0])))
            h_pos = int(player_position[1] * (new_window_width / float(self._map_shape[1])))
            pygame.draw.circle(surface, [0, 0, 0, 255], (w_pos, h_pos), 5, 0)

            for agent in objects_to_render['agents']:
                if agent.HasField('pedestrian') and objects_to_render['draw_pedestrians']:
                    if agent.id in objects_to_render['active_agents_ids']:
                        color = [255, 0, 255, 255]
                    else:
                        color = [0, 0, 255, 255]

                    agent_position = self._map.convert_to_pixel([
                        agent.pedestrian.transform.location.x,
                        agent.pedestrian.transform.location.y,
                        agent.pedestrian.transform.location.z])

                    w_pos = int(agent_position[0] * (
                            float(self._window_height) / float(self._map_shape[0])))
                    h_pos = int(agent_position[1] * (new_window_width / float(self._map_shape[1])))

                    pygame.draw.circle(surface, color, (w_pos, h_pos), 2, 0)

                if agent.HasField('traffic_light') and objects_to_render['draw_traffic_lights']:
                    if agent.id in objects_to_render['active_agents_ids']:
                        color = [255, 0, 0, 255]
                    else:
                        color = [0, 255, 0, 255]

                    agent_position = self._map.convert_to_pixel([
                        agent.traffic_light.transform.location.x,
                        agent.traffic_light.transform.location.y,
                        agent.traffic_light.transform.location.z])

                    w_pos = int(agent_position[0] * (
                            float(self._window_height) / float(self._map_shape[0])))
                    h_pos = int(
                        agent_position[1] * (new_window_width / float(self._map_shape[1])))

                    pygame.draw.circle(surface, color, (w_pos, h_pos), 3, 0)
                if agent.HasField('vehicle') and objects_to_render['draw_vehicles']:
                    if agent.id in objects_to_render['active_agents_ids']:
                        color = [255, 0, 255, 255]
                    else:
                        color = [0, 0, 255, 255]

                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])

                    w_pos = int(agent_position[0] * (
                            float(self._window_height) / float(self._map_shape[0])))
                    h_pos = int(agent_position[1] * (new_window_width / float(self._map_shape[1])))

                    pygame.draw.circle(surface, color, (w_pos, h_pos), 3, 0)

            self._display.blit(surface, (self._window_width, 0))

        self._render_iter += 1
        pygame.display.flip()

    def _draw_waypoints(self, surface, waypoints):
        """
            Draw the waypoints on the map surface.
        Args:
            surface:
            waypoints: waypoints produced by the local planner.

        Returns:

        """
        for waypoint in waypoints:
            new_window_width = \
                (float(self._window_height) / float(self._map_shape[0])) * \
                float(self._map_shape[1])

            w_pos = int(waypoint[0] * (float(self._window_height) / float(self._map_shape[0])))
            h_pos = int(waypoint[1] * (new_window_width / float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 3, 0)

    def _draw_route(self, surface, waypoints):
        """
            Draw the waypoints on the map surface.
        Args:
            surface:
            waypoints: waypoints produced by the local planner.

        Returns:

        """
        for waypoint in waypoints:
            new_window_width = \
                (float(self._window_height) / float(self._map_shape[0])) * \
                float(self._map_shape[1])

            w_pos = int(waypoint[0] * (float(self._window_height) / float(self._map_shape[0])))
            h_pos = int(waypoint[1] * (new_window_width / float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 165, 0, 255], (w_pos, h_pos), 3, 0)

    def _draw_fov(self, surface, center, player_orientation, radius, angle, color):
        new_window_width = \
            (float(self._window_height) / float(self._map_shape[0])) * \
            float(self._map_shape[1])

        w_pos = int(center[0] * (float(self._window_height) / float(self._map_shape[0])))
        h_pos = int(center[1] * (new_window_width / float(self._map_shape[1])))

        pygame.draw.arc(surface, color, (w_pos-radius/2, h_pos-radius/2, radius, radius),
                        player_orientation-angle, player_orientation + angle, int(radius/2))


    def _draw_goal_position(self, surface):
        """
            Draw the goal position on the map surface.
        Args:
            surface:

        Returns:

        """
        new_window_width = \
            (float(self._window_height) / float(self._map_shape[0])) * \
            float(self._map_shape[1])

        w_pos = int(
            self._goal_position[0] * (float(self._window_height) / float(self._map_shape[0])))
        h_pos = int(self._goal_position[1] * (new_window_width / float(self._map_shape[1])))

        pygame.draw.circle(surface, [0, 255, 255, 255], (w_pos, h_pos), 5, 0)