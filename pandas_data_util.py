import numpy as np
import pandas as pd

# Logging
# from __main__ import logger_name
logger_name = 'UTIL LOGGER'
import logging
log = logging.getLogger(logger_name)

zone_time_map = {
  'asia-east1': 'Asia/Taipei',
  'asia-east2': 'Asia/Hong_Kong',
  'asia-northeast1': 'Asia/Tokyo',
  'asia-northeast2': 'Asia/Tokyo',
  'asia-northeast3': 'Asia/Seoul',
  'asia-south1': 'Asia/Kolkata',
  'asia-southeast1': 'Asia/Singapore',
  'asia-southeast2': 'Asia/Jakarta',
  'australia-southeast1': 'Australia/Sydney',
  'europe-north1': 'Europe/Helsinki',
  'europe-west1': 'Europe/Brussels',
  'europe-west2': 'Europe/London',
  'europe-west3': 'Europe/Berlin',
  'europe-west4': 'Europe/Amsterdam',
  'europe-west6': 'Europe/Zurich',
  'northamerica-northeast1': 'America/Montreal',
  'southamerica-east1': 'America/Sao_Paulo',
  'us-central1': 'US/Central',
  'us-east1': 'US/Eastern',
  'us-east4': 'US/Eastern',
  'us-west1': 'US/Pacific',
  'us-west2': 'US/Pacific',
  'us-west3': 'US/Mountain',
  'us-west4': 'US/Pacific'
}

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

def kernel_version(kernel_release):
    kernel_split = kernel_release.split('.')
    version = float(kernel_split[0] + '.' + kernel_split[1])
    return version

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
    
  def __init__(self, filename, train, valid, horizon, window, normalize = 0, query='vm_1_gce_network_tier == "premium"'):

    df = pd.read_csv(filename, dtype={"ping_average_latency": float, 
                                      "iperf_throughput_1_thread": float,
                                      "iperf_throughput_32_threads": float})

    ################# Data prep #################
    df[['vm_1_gce_network_tier',
        'vm_2_gce_network_tier']] = df[['vm_1_gce_network_tier',
                                        'vm_2_gce_network_tier']].fillna(value='premium')
    df[['tcp_max_receive_buffer']] = df[['tcp_max_receive_buffer']].fillna(value=6291456)
    # df = df.set_index('thedate')
    
    # Filter the data to what we want
    
    df = df[df['iperf_throughput_1_thread'].notnull()]
    df = df[df['ping_average_latency'].notnull()]
    df = df[df['sending_zone'].notnull()]
    df = df[df['receiving_zone'].notnull()]
    # df = df[df['vm_1_cloud'].notnull()]
    # df = df[df['vm_2_cloud'].notnull()]
    df[['pandas_datetime']] = df[['thedate']].apply(pd.to_datetime)
    
    # Reduce dataset to just recent where it is more stable
    df = df[df['pandas_datetime'] > '03/03/2022']
    df.reset_index(inplace=True,drop=True)
    
    # Get local time
    df['sending_zone_datetime'] = df.apply(lambda row: row['pandas_datetime'].tz_convert(zone_time_map[row['sending_zone'][:-2]]), axis=1)
    df['receiving_zone_datetime'] = df.apply(lambda row: row['pandas_datetime'].tz_convert(zone_time_map[row['receiving_zone'][:-2]]), axis=1)

    # Remove timezone data
    df['sending_zone_datetime'] = df.apply(lambda row: pd.to_datetime(row['sending_zone_datetime'].tz_localize(None).strftime("%Y/%m/%d, %H:%M:%S")), axis=1)
    df['receiving_zone_datetime'] = df.apply(lambda row: pd.to_datetime(row['receiving_zone_datetime'].tz_localize(None).strftime("%Y/%m/%d, %H:%M:%S")), axis=1)
    
    # extract hour of the day and day of the week
    df['sending_zone_day'] = df['sending_zone_datetime'].dt.dayofweek
    df['receiving_zone_day'] = df['receiving_zone_datetime'].dt.dayofweek
    df['sending_zone_hour'] = df['sending_zone_datetime'].dt.hour
    df['receiving_zone_hour'] = df['receiving_zone_datetime'].dt.hour

    # convert hour of the day and day of the week into sin and cos components because they are cyclical
    df['sending_zone_day_cos'] = np.cos(2*np.pi*df['sending_zone_day']/7)
    df['sending_zone_day_sin'] = np.sin(2*np.pi*df['sending_zone_day']/7)

    df['receiving_zone_day_cos'] = np.cos(2*np.pi*df['receiving_zone_day']/7)
    df['receiving_zone_day_sin'] = np.sin(2*np.pi*df['receiving_zone_day']/7)

    df['sending_zone_hour_cos'] = np.cos(2*np.pi*df['sending_zone_hour']/24)
    df['sending_zone_hour_sin'] = np.sin(2*np.pi*df['sending_zone_hour']/24)

    df['receiving_zone_hour_cos'] = np.cos(2*np.pi*df['receiving_zone_hour']/24)
    df['receiving_zone_hour_sin'] = np.sin(2*np.pi*df['receiving_zone_hour']/24)
    
    df['kernel_version'] = df['vm_1_kernel_release'].apply(kernel_version)
    df.sort_values(by=['iperf_timestamp', 'run_uri'], inplace=True, ascending=True)
    df = df.set_index('thedate')
    one_hot_machine_type = pd.get_dummies(df['vm_1_machine_type'], dtype='float32')
    df = df.join(one_hot_machine_type)
    one_hot_ip_type = pd.get_dummies(df['ip_type'], dtype='float32')
    df = df.join(one_hot_ip_type)
    
    df['vm_1_os_info_trunc'] = df['vm_1_os_info'].str.extract(r'(\w+\s+\d+.\d+).\d+\s+\w+')
    
    one_hot_congestion_control = pd.get_dummies(df['tcp_congestion_control'], dtype='float32')
    df = df.join(one_hot_congestion_control)
    
    # Don't do the VM os type, that's too much
    # one_hot_vm_1_os_info_trunc = pd.get_dummies(df['vm_1_os_info_trunc'], dtype='float32')
    # df = df.join(one_hot_vm_1_os_info_trunc)
    
    
    print(df.columns)
    
    df['group_mean'] = df.groupby(['sending_zone',
                       'receiving_zone',
                       'tcp_max_receive_buffer',
                       'vm_1_machine_type',
                       'ip_type',
                       'tcp_congestion_control',
                       'vm_1_os_info_trunc'
                        ])['iperf_throughput_32_threads'].transform('mean')
    
    df['group_stddev'] = df.groupby(['sending_zone',
                           'receiving_zone',
                           'tcp_max_receive_buffer',
                           'vm_1_machine_type',
                           'ip_type',
                           'tcp_congestion_control',
                           'vm_1_os_info_trunc'
                            ])['iperf_throughput_32_threads'].transform('std')
    
    
    df['group_id'] = df.groupby(['sending_zone',
                           'receiving_zone',
                           'tcp_max_receive_buffer',
                           'vm_1_machine_type',
                           'ip_type',
                           'tcp_congestion_control',
                           'vm_1_os_info_trunc'
                            ])['iperf_throughput_32_threads'].ngroup()
    # TODO rethink how are are normalizing data
    # maybe we want to do it per group?
    # normalize data
    mean = df['ping_average_latency'].mean(axis=0)
    df[['ping_average_latency']] = df[['ping_average_latency']] - mean
    std = df['ping_average_latency'].std(axis=0)
    df[['ping_average_latency']] /= std

    mean = df['iperf_throughput_1_thread'].mean(axis=0)
    df[['iperf_throughput_1_thread']] = df[['iperf_throughput_1_thread']] - mean
    std = df['iperf_throughput_1_thread'].std(axis=0)
    df[['iperf_throughput_1_thread']] /= std
    
    mean = df['iperf_throughput_32_threads'].mean(axis=0)
    self.iperf_32_thread_mean = mean
    df[['iperf_throughput_32_threads']] = df[['iperf_throughput_32_threads']] - mean
    std = df['iperf_throughput_32_threads'].std(axis=0)
    self.iperf_32_thread_std = std
    df[['iperf_throughput_32_threads']] /= std

    mean = df['tcp_max_receive_buffer'].mean(axis=0)
    df[['tcp_max_receive_buffer']] = df[['tcp_max_receive_buffer']] - mean
    std = df['tcp_max_receive_buffer'].std(axis=0)
    df[['tcp_max_receive_buffer']] /= std
    
    mean = df['kernel_version'].mean(axis=0)
    df[['kernel_version']] = df[['kernel_version']] - mean
    std = df['kernel_version'].std(axis=0)
    df[['kernel_version']] /= std
    
#     mean = df['n1-standard-2'].mean(axis=0)
#     df[['n1-standard-2']] = df[['n1-standard-2']] - mean
#     std = df['n1-standard-2'].std(axis=0)
#     df[['n1-standard-2']] /= std
    
#     mean = df['n1-standard-16'].mean(axis=0)
#     df[['n1-standard-16']] = df[['n1-standard-16']] - mean
#     std = df['n1-standard-16'].std(axis=0)
#     df[['n1-standard-16']] /= std
    
#     mean = df['external'].mean(axis=0)
#     df[['external']] = df[['external']] - mean
#     std = df['external'].std(axis=0)
#     df[['external']] /= std
    
#     mean = df['internal'].mean(axis=0)
#     df[['internal']] = df[['internal']] - mean
#     std = df['internal'].std(axis=0)
#     df[['internal']] /= std
    
#     mean = df['bbr'].mean(axis=0)
#     df[['bbr']] = df[['bbr']] - mean
#     std = df['bbr'].std(axis=0)
#     df[['bbr']] /= std
    
#     mean = df['cubic'].mean(axis=0)
#     df[['cubic']] = df[['bbr']] - mean
#     std = df['cubic'].std(axis=0)
#     df[['cubic']] /= std
    
    df = df.query(query)
    
    self.columns   = ['pandas_datetime',
                      'iperf_throughput_1_thread',
                      'iperf_throughput_32_threads',
                      'ping_average_latency',
                      'tcp_max_receive_buffer',
                      'sending_zone_day_cos',
                      'sending_zone_day_sin',
                      'receiving_zone_day_cos',
                      'receiving_zone_day_sin',
                      'sending_zone_hour_cos',
                      'sending_zone_hour_sin',
                      'receiving_zone_hour_cos',
                      'receiving_zone_hour_sin', 
                      'kernel_version',
                      'group_mean',
                      'group_stddev'
                      # 'bbr', 'cubic',
                      # 'n1-standard-16', 'n1-standard-2',
                      # 'external', 'internal',
                      # 'Ubuntu 14.04', 'Ubuntu 16.04', 'Ubuntu 18.04', 'Ubuntu 20.04',
                      # 'vm_1_machine_type',
                      # 'ip_type',
                      # 'tcp_congestion_control',
                      # 'vm_1_os_info_trunc'
                     ]

    # sort and group
    gb = df.groupby(['sending_zone',
                     'receiving_zone',
                     'tcp_max_receive_buffer',
                     'vm_1_machine_type',
                     'ip_type',
                     'tcp_congestion_control',
                     'vm_1_os_info_trunc'], 
                    as_index=False)[self.columns]
    
    self.groups = list(gb.groups)

    self.rawdata = gb

    log.debug("End reading data")

    self.w         = window
    self.h         = horizon
    self.m         = self.rawdata.get_group(self.groups[0]).shape[-1]
    self.data      = None
    self.n         = None
    self.normalize = normalize
    self.scale     = np.ones(self.m)
    

    # TODO normalize before split into groups
    self.normalize_data(normalize)
    # self.split_data(train, valid)

    split_results = self.pandas_group_split_and_batchify(self.data, train, 
                                                         valid, 1.0-train-valid,
                                                         self.w, self.h, step=1)


    self.train = split_results[0]
    self.valid = split_results[1]
    self.test  = split_results[2]
    
  def save_data(self, filename):
    pass
  # TODO implement
        
        
  def normalize_data(self, normalize):
    log.debug("Normalize: %d", normalize)

    if normalize == 0: # do not normalize
        self.data = self.rawdata
    
    if normalize == 1: # same normalisation for all timeseries
        self.data = self.rawdata / np.max(self.rawdata)
    
    if normalize == 2: # normalize each timeseries alone. This is the default mode
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
    
    # Max time delta = 30 hours
    max_time_delta_sec = 30 * 60 * 60
    min_time_delta_sec = 20 * 60 * 60

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

      for j,row in enumerate(rows):
        low_index = rows[j] - window
        high_index = rows[j]
        sample = group_arrays[current_group]['values'][low_index:high_index:step]
        target = group_arrays[current_group]['values'][rows[j] + horizon]
        # print("j: ", rows[j])
        # print(sample)
        # print(target)
        
        # Check dates of sample to make sure there are no gaps
        bad_sample = False
        for sample_index in range(1, len(sample)):
          difference = sample[sample_index][0] - sample[sample_index-1][0]
          delta_in_seconds = difference.total_seconds()
          if delta_in_seconds > max_time_delta_sec:
            bad_sample = True
            break
          elif delta_in_seconds < min_time_delta_sec:
            bad_sample = True
            break
        if bad_sample:
          continue

        sample_list.append(sample)
        target_list.append(target)
    
    if(len(sample_list) == 0):
      print("No samples in list. Try a smaller horizon or window")

    return [np.array(sample_list), np.array(target_list)]