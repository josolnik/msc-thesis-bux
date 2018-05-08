def get_values():

  conn_info = {'host': 'vertica.getbux.com',
              'port': 5433,
              'user': 'josolnik',
              'password': 'CXw!CHeDa@N6hGQ8',
              'database': 'buxdwh',
              # 10 minutes timeout on queries
              'read_timeout': 600,
              # default throw error on invalid UTF-8 results
              'unicode_error': 'strict',
              # SSL is disabled by default
              'ssl': False,
              'connection_timeout': 5
              # connection timeout is not enabled by default
           }
  return conn_info