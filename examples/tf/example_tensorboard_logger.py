import tensorflow as tf

from garage.misc import logger, tabular
from garage.misc.logger import TensorBoardOutput

logger.add_output(TensorBoardOutput("data/local/histogram_example"))
N = 400
for i in range(N):
    sess = tf.Session()
    sess.__enter__()
    k_val = i / float(N)
    # logger.record_histogram_by_type('gamma', key='gamma', alpha=k_val)
    # logger.record_histogram_by_type(
    #     'normal', key='normal', mean=5 * k_val, stddev=1.0)
    # logger.record_histogram_by_type('poisson', key='poisson', lam=k_val)
    # logger.record_histogram_by_type(
    #     'uniform', key='uniform', maxval=k_val * 10)
    tabular.record_tabular("app", k_val)
    logger.log(("gass", k_val), record='histogram')
    logger.dump(TensorBoardOutput, step=i)
