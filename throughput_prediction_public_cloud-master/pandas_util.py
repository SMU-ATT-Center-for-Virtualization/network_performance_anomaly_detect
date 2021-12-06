#Modified version of data_util from LSTNet

import numpy as np
import pandas as pd

# Logging
from __main__ import logger_name
import logging
log = logging.getLogger(logger_name)

def min_percent_index(num, percent, offset):
    min_index = round((num-1)*percent) + offset
    if min_index >= num:
        min_index = num-1
    return min_index

def max_percent_index(num, percent, offset):
    max_index = round((num-1)*percent) + offset
    if max_index >= num:
        max_index = num-1
    return max_index

class DataUtil(object):
  #
  # This class contains data specific information.
  # It does the following:
  #  - Read data from file
  #  - Normalise it
  #  - Split it into train, dev (validation) and test
  #  - Create X and Y for each of the 3 sets (train, dev, test) according to the following:
  #    Every sample (x, y) shall be created as follows:
  #     - x --> window number of values
  #     - y --> one value that is at horizon in the future i.e. that is horizon away past the last value of x
  #    This way X and Y will have the following dimensions:
  #     - X [number of samples, window, number of multivariate time series]
  #     - Y [number of samples, number of multivariate time series]
    
  def __init__(self, filename, train, valid, horizon, window, normalise = 0, dependent_variable='iperf_throughput_1_thread'):

    df = pd.read_csv(filename, dtype={"ping_average_latency": float, 
                                       dependent_variable: float})

    ################# Data prep #################
    df[['vm_1_gce_network_tier',
        'vm_2_gce_network_tier']] = df[['vm_1_gce_network_tier',
                                        'vm_2_gce_network_tier']].fillna(value='premium')
    df[['tcp_max_receive_buffer']] = df[['tcp_max_receive_buffer']].fillna(value=6291456)
    df = df.set_index('thedate')
    query = 'vm_1_machine_type == "n1-standard-16" and ip_type == "internal" and vm_1_gce_network_tier == "premium"'
    df = df.query(query)
    df = df[df[dependent_variable].notnull()]
    df = df[df['sending_zone'].notnull()]
    df = df[df['receiving_zone'].notnull()]

    # normalize data
    self.normalize_mean = []
    self.normalize_std = []

    mean = df[dependent_variable].mean(axis=0)
    df[[dependent_variable]] = df[[dependent_variable]] - mean
    std = df[dependent_variable].std(axis=0)
    df[[dependent_variable]] /= std
    self.normalize_mean.append(mean)
    self.normalize_std.append(std)

    mean = df['tcp_max_receive_buffer'].mean(axis=0)
    df[['tcp_max_receive_buffer']] = df[['tcp_max_receive_buffer']] - mean
    std = df['tcp_max_receive_buffer'].std(axis=0)
    df[['tcp_max_receive_buffer']] /= std
    self.normalize_mean.append(mean)
    self.normalize_std.append(std)

    mean = df['ping_average_latency'].mean(axis=0)
    df[['ping_average_latency']] = df[['ping_average_latency']] - mean
    std = df['ping_average_latency'].std(axis=0)
    df[['ping_average_latency']] /= std
    self.normalize_mean.append(mean)
    self.normalize_std.append(std)


    print(self.normalize_mean[0])
    print(self.normalize_std[0])

    # sort and group
    df = df.sort_values(['sending_zone', 'receiving_zone', 'tcp_max_receive_buffer', 'thedate']).reset_index()
    gb = df.groupby(['sending_zone',
                     'receiving_zone',
                     'tcp_max_receive_buffer'], 
                     as_index=False)[[
                                      # 'thedate',
                                      # 'sending_zone',
                                      # 'receiving_zone',
                                      dependent_variable,
                                      'ping_average_latency',
                                      'tcp_max_receive_buffer'
                                      ]]
    self.groups = list(gb.groups)

    self.rawdata = gb

    log.debug("End reading data")

    self.w         = window
    self.h         = horizon
    self.m         = self.rawdata.get_group(self.groups[0]).shape[-1]
    self.data      = None
    self.n         = None
    self.normalise = normalise
    self.scale     = np.ones(self.m)

    # TODO normalize before split into groups
    self.normalise_data(normalise)
    # self.split_data(train, valid)

    split_results = self.pandas_group_split_and_batchify(self.data, train, 
                                                         valid, 1.0-train-valid,
                                                         self.w, self.h, step=1)


    self.train = split_results[0]
    self.valid = split_results[1]
    self.test  = split_results[2]    
        
        
  def normalise_data(self, normalise):
    log.debug("Normalise: %d", normalise)

    if normalise == 0: # do not normalise
        self.data = self.rawdata
    
    if normalise == 1: # same normalisation for all timeseries
        self.data = self.rawdata / np.max(self.rawdata)
    
    if normalise == 2: # normalise each timeseries alone. This is the default mode
        for i in range(self.m):
            self.scale[i] = np.max(np.abs(self.rawdata[:, i]))
            self.data[:, i] = self.rawdata[:, i] / self.scale[i]


  def split_data(self, train, valid):
    log.info("Splitting data into training set (%.2f), validation set (%.2f) and testing set (%.2f)", train, valid, 1 - (train + valid))

    train_set = range(self.w + self.h - 1, int(train * self.n))
    valid_set = range(int(train * self.n), int((train + valid) * self.n))
    test_set  = range(int((train + valid) * self.n), self.n)
    
    self.train = self.get_data(train_set)
    self.valid = self.get_data(valid_set)
    self.test  = self.get_data(test_set)



  def pandas_group_split_and_batchify(self, data, train_percentage, valid_percentage, test_percentage,
                                      window, horizon, step=1):
    try:
      assert 1.0 >= train_percentage >= 0.0
      assert 1.0 >= valid_percentage >= 0.0
      assert 1.0 >= test_percentage >= 0.0
      assert (train_percentage + valid_percentage + test_percentage) <= 1.0
    except AssertionError:
      handleAssertError()
      return

    log.info("Splitting data into training set (%.2f), validation set (%.2f) and testing set (%.2f)", train_percentage, valid_percentage, test_percentage)

    # train_set = range(self.P+self.h-1, train)
    # valid_set = range(train, valid)
    # test_set = range(valid, self.n)
    valid_min = train_percentage
    valid_max = train_percentage + test_percentage
    test_min = train_percentage + valid_percentage
    test_max = train_percentage + valid_percentage + test_percentage
    train = self.pandas_group_batchify(data,window,horizon, 
                                      min_percent=0.0, 
                                      max_percent=train_percentage)
    valid = self.pandas_group_batchify(data, window, horizon, 
                                      min_percent=valid_min, 
                                      max_percent=valid_max,
                                      min_offset = 1)
    test = self.pandas_group_batchify(data, window, horizon,
                                     min_percent=test_min, 
                                     max_percent=test_max,
                                     min_offset = 1)
    return [train,valid,test]


        
  def get_data(self, rng):
    n = len(rng)
    
    X = np.zeros((n, self.w, self.m))
    Y = np.zeros((n, self.m))
    
    for i in range(n):
      end   = rng[i] - self.h + 1
      start = end - self.w
      
      X[i,:,:] = self.data[start:end, :]
      Y[i,:]   = self.data[rng[i],:]
    
    return [X, Y]


  # Note this will return empty arrays if window/horizon are too large for dataset
  # TODO allow to specify columns to output
  def pandas_group_batchify(self, data, window, horizon, min_percent=0.0, max_percent=1.0,
                            min_offset=0, max_offset=0, step=1, cols=[]):
    """Takes pandas group object, returns windows of data in specified range
    
    Creates generator that returns batches from mutiple groups of data, keeping each sample
    to be from only a single batch. This means you need to check your delay, and window to
    make sure that each group can handle at least one sample/target of the specified size. If not,
    that group will be skipped and won't be present in the resultant dataset.

    You can also split up the dataset by percentages. 
    ex. train_gen = pandas_group_generator(data, 10, 5, min_percentage=0, max_percentage=0.40)
        test_gen = pandas_group_generator(data, 10, 5, min_percentage=0.40, min_offset=1)

    TODO be able to split up by groups?
    
    Arguments:
      data {np array} -- The original array of floating point data
      window {int} -- How many timesteps back should our input data go.
      horizon {int} -- How many timesteps in the future should our target be.
                       0 means predict this value, given previous <window> values
    
    Keyword Arguments:
      min_percent {number} -- The lower percentage (default: {0.0})
      max_percent {number} -- [description] (default: {1.0})
      min_offset {number} -- [description] (default: {0})
      max_offset {number} -- [description] (default: {0})
      shuffle {bool} -- [description] (default: {False})
      batch_size {number} -- [description] (default: {128})
      step {number} -- [description] (default: {1})

    returns samples and targets for each sample
    """
    assert 0 <= min_percent <= 1
    assert 0 <= max_percent <= 1

    group_list = list(data.groups)

    m = data.get_group(group_list[0]).shape[-1]

    group_arrays = {}
    for g in group_list:
      group_arrays[g] = {}
      group_arrays[g]['metadata'] = ""
      group_arrays[g]['values'] = data.get_group(g).values.tolist()

    total_steps = 0
    for g in group_list:
      size = len(group_arrays[g]['values'])
      min_index = min_percent_index(size, min_percent, min_offset)
      max_index = max_percent_index(size, max_percent, max_offset)
  #     print(min_index)
  #     print(max_index)
      required_steps = (max_index - min_index - window)
  #     print(required_steps)
      total_steps += required_steps
      
    sample_list = []
    target_list = []

    for group_list_position in range(0,len(group_list)):
      # cycle through groups in dataframe
      current_group = group_list[group_list_position]
      min_index = min_percent_index(len(group_arrays[current_group]['values']),
                                    min_percent, min_offset)
      max_index = max_percent_index(len(group_arrays[current_group]['values']),
                                    max_percent, max_offset)
      max_index = max_index - horizon
      i = min_index + window

      rows = np.arange(i, max_index+1)
      # print("i: ", i)
      # print("max index: ", max_index)
      # print(rows)
      local_sample_list = []
      local_target_list = []
      for j,row in enumerate(rows):
        low_index = rows[j] - window
        high_index = rows[j]
        sample = group_arrays[current_group]['values'][low_index:high_index:step]
        target = group_arrays[current_group]['values'][rows[j] + horizon]
        # print("j: ", rows[j])
        # print(sample)
        # print(target)
        sample_list.append(sample)
        target_list.append(target)

        local_sample_list.append(sample)
        local_target_list.append(target)

      #if max_percent==1.0:

        
        #filename = 'C:\\Users\\Troy\\Desktop\\throughput_prediction_public_cloud-master\\bq-results.csv' + str(group_list_position)
        
        #filename = 'numpy_group_test_data/32_stream_test_sample_group_' + str(group_list_position)
        #np.save(filename, np.array(local_sample_list, dtype="float32"))

        #filename = 'C:\\Users\\Troy\\Desktop\\throughput_prediction_public_cloud-master\\bq-results.csv' + str(group_list_position)
        #filename = 'numpy_group_test_data/32_stream_test_target_group_' + str(group_list_position)
        #np.save(filename, np.array(local_target_list, dtype="float32"))

    
    if(len(sample_list) == 0):
      print("No samples in list. Try a smaller horizon or window")

    return [np.array(sample_list, dtype="float32"), np.array(target_list, dtype="float32")]
