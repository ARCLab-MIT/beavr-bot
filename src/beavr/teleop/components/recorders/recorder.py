import logging

from beavr.teleop.components import Component

logger = logging.getLogger(__name__)


class Recorder(Component):
    def _add_metadata(self, datapoints):
        self.metadata = dict(
            file_name = self._recorder_file_name,
            num_datapoints = datapoints,
            record_start_time = self.record_start_time,
            record_end_time = self.record_end_time,
            record_duration = self.record_end_time - self.record_start_time,
            record_frequency = datapoints / (self.record_end_time - self.record_start_time)
        )

    def _display_statistics(self, datapoints):
        logger.info('Saving data to {}'.format(self._recorder_file_name))
        logger.info('Number of datapoints recorded: {}.'.format(datapoints))
        logger.info('Data record frequency: {}.'.format(datapoints / (self.record_end_time - self.record_start_time)))