import os
import json
import numpy as np
import pandas as pd
import psycopg2

FLAGS = None
ROOT = os.path.join(os.path.dirname(__file__), '../')

config_file = os.path.join(ROOT, 'config/config.json')
pickle_file = os.path.join(ROOT, 'cache', 'telemetry.npy')


def main():

    with open(config_file) as conf:
        config = json.load(conf)
    conn_config = config['database']

    try:
        conn = psycopg2.connect(host=conn_config['host'],
                                port=conn_config['port'],
                                user=conn_config['user'],
                                password=conn_config['password'],
                                dbname=conn_config['dbname'])

        # get and pickle raw data
        get_telemetry(conn)

        extractor = FeatureLabelExtractor(pickle_file)
        X = extractor.get_sequential_features

        for col in X.columns.values:
            print(sum(pd.isnull(X[col])))
            print(X[col])

    finally:
        conn.close()


def get_telemetry(conn):
    curs = conn.cursor()

    query = """
    SELECT created_at as date, data as telemetry, model, os, osv, real_pump
    FROM total_telemetry_data
    """

    curs.execute(query)
    data = list(curs.fetchall())

    np.save(os.path.join(ROOT, 'cache', 'telemetry.npy'), data)

    curs.close()


class FeatureLabelExtractor:

    """
    Get the features (sequential or not) and labels from pickle.

    Parameters:
    ----------
    pickle_file: the .npy file to be unpacked


    Attributes:
    ----------


    """

    def __init__(self, pickle_file):

        self._raw_data = np.load(pickle_file)
        self._raw_telemetry = []
        self._telemetry = {'8594C654-6565-4DBC-9FA6-BEC41B929609_1_1': [],
                          '8594C654-6565-4DBC-9FA6-BEC41B929609_1_2': [],
                          '8594C654-6565-4DBC-9FA6-BEC41B929609_2_1': [],
                          '8594C654-6565-4DBC-9FA6-BEC41B929609_2_2': [],
                          '8594C654-6565-4DBC-9FA6-BEC41B929609_3_1': [],
                          '8594C654-6565-4DBC-9FA6-BEC41B929609_3_2': [],
                          '8594C654-6565-4DBC-9FA6-BEC41B929609_3_3': [],
                          '1192EF5B-6CA3-4ACA-B632-C00BC1CC703C_1_1': [],
                          '1192EF5B-6CA3-4ACA-B632-C00BC1CC703C_1_2': [],
                          '1192EF5B-6CA3-4ACA-B632-C00BC1CC703C_2_1': [],
                          '1192EF5B-6CA3-4ACA-B632-C00BC1CC703C_2_2': [],
                          '1192EF5B-6CA3-4ACA-B632-C00BC1CC703C_3_1': [],
                          '1192EF5B-6CA3-4ACA-B632-C00BC1CC703C_3_2': [],
                          '1192EF5B-6CA3-4ACA-B632-C00BC1CC703C_3_3': [],
                          'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_1_1': [],
                          'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_1_2': [],
                          'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_2_1': [],
                          'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_2_2': [],
                          'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_3_1': [],
                          'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_3_2': [],
                          'F74B56F7-5F12-413B-BF5E-DF09CC7E5C33_3_3': []}
        self._sequential_features = []
        self._features = []
        self._label = []

    @property
    def get_labels(self):

        self._label = np.asarray(list(map(lambda x: x[:][5], self._raw_data)))
        return self._label

    @property
    def get_features(self):

        dates = np.asarray(list(map(lambda x: x[:][0], self._raw_data)))
        model = np.asarray(list(map(lambda x: x[:][2], self._raw_data)))
        o_s = np.asarray(list(map(lambda x: x[:][3], self._raw_data)))
        osv = np.asarray(list(map(lambda x: x[:][4], self._raw_data)))

        self._features = pd.DataFrame(data={'dates': dates,
                                            'model': model,
                                            'os': o_s,
                                            'os_version': osv})
        return self._features

    @property
    def get_sequential_features(self):

        # print(self.raw_data)
        self._raw_telemetry = np.asarray(list(map(lambda x: x[:][1], self._raw_data)))

        # for each test
        for line in range(len(self._raw_telemetry)):

            # get the list of detected beacons in the test
            detected_beacons = []
            for beacon in range(len(self._raw_telemetry[line])):
                key = self.get_uuid_major_minor(line, beacon)
                detected_beacons.append(key)

            # for each possible beacon append  beacon data if detected
            # otherwise append empty list
            for beacon in self._telemetry.keys():

                if beacon in detected_beacons:
                    # find that beacon in raw_telemetry
                    for detected_beacon in range(len(self._raw_telemetry[line])):

                        key = self.get_uuid_major_minor(line, detected_beacon)
                        if key == beacon:
                            self._telemetry[beacon].append(self._raw_telemetry[line][detected_beacon]['raw_data'])
                            break
                        else:
                            pass
                else:
                    # self._telemetry[beacon].append(np.nan)
                    self._telemetry[beacon].append([])


        # make Pandas DataFrame
        self._sequential_features = pd.DataFrame(data=self._telemetry)
        return self._sequential_features

    def get_uuid_major_minor(self, line, beacon):

        uuid = self._raw_telemetry[line][beacon]['beacon']['uuid']
        major = self._raw_telemetry[line][beacon]['beacon']['major']
        minor = self._raw_telemetry[line][beacon]['beacon']['minor']
        key = uuid + '_' + str(major) + '_' + str(minor)
        return key


if __name__ == '__main__':
    main()
