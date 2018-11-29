import pickle
from os import path as osp


class Snapshotter(object):
    def __init__(self):
        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1

    def set_snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self, ):
        return self._snapshot_dir

    def get_snapshot_mode(self, ):
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def get_snapshot_gap(self, ):
        return self._snapshot_gap

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def save_extra_data(self, data, file_name='extra_data.pkl', mode='joblib'):
        """
        Data saved here will always override the last entry
        :param data: Something pickle'able.
        """
        if self._snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
            if mode == 'joblib':
                import joblib
                joblib.dump(data, file_name, compress=3)
            elif mode == 'pickle':
                pickle.dump(data, open(file_name, "wb"))
            else:
                raise ValueError("Invalid mode: {}".format(mode))
            return file_name

        return None

    def save_itr_params(self, itr, params):
        if self._snapshot_dir:
            if self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == "gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir,
                                         'itr_%d.pkl' % itr)
                    pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == "gap_and_last":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir,
                                         'itr_%d.pkl' % itr)
                    pickle.dump(params, open(file_name, "wb"))
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise NotImplementedError


snapshotter = Snapshotter()
