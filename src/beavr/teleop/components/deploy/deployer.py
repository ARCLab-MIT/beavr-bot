import pickle

from beavr.teleop.components import Component
from beavr.teleop.utils.network import create_response_socket
from beavr.teleop.utils.timer import FrequencyTimer
from beavr.teleop.configs.constants import network
from beavr.teleop.utils.instantiator import instantiate_from_target
import logging

logger = logging.getLogger(__name__)

class DeployServer(Component):
    """
    DeployServer is a component that deploys the robot to the desired state.
    """
    def __init__(self, configs):
        self.configs = configs
        
        # Initializing the camera subscribers
        self._init_robot_subscribers()

        # Initializing the sensor subscribers 
        if configs.use_sensor:
            self._init_sensor_subscribers()

        self.deployment_socket = create_response_socket(
            host = self.configs.host_address,
            port = self.configs.deployment_port
        )

        self.timer = FrequencyTimer(network.DEPLOY_FREQ)

    def _init_robot_subscribers(self):
        robot_controllers = instantiate_from_target(self.configs.robot.controllers)
        self._robots = dict()
        for robot in robot_controllers:
            self._robots[robot.name] = robot

    def _init_sensor_subscribers(self):
        xela_controllers = instantiate_from_target(self.configs.robot.xela_controllers)
        self._sensors = dict()
        self._sensors['xela'] = xela_controllers[0] # There is only 1 controller

    def _perform_robot_action(self, robot_action_dict):
        try:

            # Kinova should be applied earlier than allegro
            robot_order = ['franka', 'allegro']

            for robot in robot_order:
                if robot in robot_action_dict.keys():
                    if robot not in self._robots.keys():
                        logger.error('Robot: {} is an illegal argument.'.format(robot))
                        return False
                    if robot == 'kinova' or robot == 'franka':
                        # We use cartesian coords with kinova and not the joint angles
                        logger.info('Moving the arm in cartesian coords! to: {}'.format(robot_action_dict[robot]))
                        if robot == 'franka': # Move the arm with a given duration
                            # self._robots[robot].move_coords(robot_action_dict[robot], duration=1/DEPLOY_FREQ)
                            self._robots[robot].arm_control(robot_action_dict[robot])
                        else:
                            self._robots[robot].move_coords(robot_action_dict[robot])

                    else: 
                        logger.info('Moving allegro to given angles')
                        self._robots[robot].move(robot_action_dict[robot])
                    logger.info('Applying action {} on robot: {}'.format(robot_action_dict[robot], robot))
            return True
        except Exception as e:
            logger.error(f'robot: {robot} failed executing in perform_robot_action: {e}')
            return False

    def _get_robot_states(self):
        data = dict()
        for robot_name in self._robots.keys():
            # Get the cartesian state for kinova
            if robot_name == 'kinova' or robot_name == 'franka':
                data[robot_name] = self._robots[robot_name].get_cartesian_position() # Get the position only
                # print(f'data[{robot_name}].shape: {data[robot_name].shape}')
            else: # allegro
                data[robot_name] = self._robots[robot_name].get_joint_state()
                # print(f'data[{robot_name}].shape: {data[robot_name].shape}')
        return data

    def _get_sensor_states(self):
        data = dict() 
        for sensor_name in self._sensors.keys():
            data[sensor_name] = self._sensors[sensor_name].get_sensor_state() # For xela this will be dict {sensor_values: [...], timestamp: [...]}

        return data

    def _send_robot_state(self):
        robot_states = self._get_robot_states()
        logger.info('robot_states: {}'.format(robot_states))
        self.deployment_socket.send(pickle.dumps(robot_states, protocol = -1))

    def _send_sensor_state(self):
        sensor_states = self._get_sensor_states()
        self.deployment_socket.send(pickle.dumps(sensor_states, protocol = -1))

    def _send_both_state(self):
        combined = dict()
        robot_states = self._get_robot_states()
        combined['robot_state'] = robot_states 
        if self.configs.use_sensor:
            sensor_states = self._get_sensor_states()
            combined['sensor_state'] = sensor_states
        self.deployment_socket.send(pickle.dumps(combined, protocol = -1))

    def stream(self):
        self.notify_component_start('robot deployer')
        # self.visualizer_process.start()
        while True:
            try:
                logger.info('\nListening')
                self.timer.start_loop()
                robot_action = pickle.loads(self.deployment_socket.recv())

                if robot_action == 'get_state':
                    logger.info('Requested for robot state information.')
                    self._send_robot_state()
                    continue

                if robot_action == 'get_sensor_state':
                    logger.info('Requested for sensor information.')
                    self._send_sensor_state()
                    continue

                success = self._perform_robot_action(robot_action)
                logger.info('success: {}'.format(success))
                # More accurate sleep
                
                self.timer.end_loop()

                if success:
                    logger.info('Before sending the states')
                    self._send_both_state()
                    logger.info('Applied robot action.')
                else:
                    self.deployment_socket.send("Command failed!")
            except Exception as e:
                logger.error(f'Illegal values passed. Terminating session: {e}')
                break

        # self.visualizer_process.join()
        logger.info('Closing robot deployer component.')
        self.deployment_socket.close()