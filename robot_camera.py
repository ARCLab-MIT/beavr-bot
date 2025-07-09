import hydra
from beavr.teleop.components import RealsenseCameras


@hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'realsense')
def main(configs):
    cameras = RealsenseCameras(configs)
    processes = cameras.get_processes()

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()