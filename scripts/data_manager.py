import os
import json
import numpy as np
import pandas as pd

import psycopg2


def getTelemetry(conn): 

	curs = conn.cursor()

	query = '''
	SELECT created_at as date, data as telemetry, model, os, osv, real_pump
	FROM total_telemetry_data
	'''

	curs.execute(query)
	data = list(curs.fetchall())

	curs.close()

	return


