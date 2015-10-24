from flask import Flask
from flask.ext.cors import CORS
app = Flask(__name__)
CORS(app)
import time
import os

LOG_FILE = 'lettersketch.log'
print("Logging to",LOG_FILE)

@app.route('/<event>/<letter>')
def record_event(event,letter):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE,'a') as f:
            f.write('{},{},{}\n'.format(time.time(),
                event,
                letter))
    else:
        with open(LOG_FILE,'w') as f:
            f.write('time_utc,event,letter\n')
            f.write('{},{},{}\n'.format(time.time(),
                event,
                letter))

    return 'Hello World!'

if __name__ == '__main__':
    app.run()
