import os
import json
import numpy as np
import pandas as pd
import psycopg2


ROOT = os.path.join(os.path.dirname(__file__), '../')
FLAGS = None


def main():

    config_file = os.path.join(ROOT, 'config/config.json')

    with open(config_file) as conf:
        config = json.load(conf)
    conn_config = config['database']

    try:
        conn = psycopg2.connect(host=conn_config['host'],
                                port=conn_config['port'],
                                user=conn_config['user'],
                                password=conn_config['password'],
                                dbname=conn_config['dbname'])

        telemetry = getTelemetry(conn)

    finally:
        conn.close()



def getTelemetry(conn): 

    curs = conn.cursor()

    query = '''
    SELECT created_at as date, data as telemetry, model, os, osv, real_pump
    FROM total_telemetry_data
    '''

    curs.execute(query)
    data = list(curs.fetchall())

    np.save(os.path.join(ROOT, 'cache', 'telemetry.npy'), data)
    data = data[0]
    print(data[0])
    print(data[1][0])
    print(data[1][1])

    # idfas = np.asarray(list(map(lambda x: x[:][0], data)))

    curs.close()

    return


if __name__ == '__main__':
    main()