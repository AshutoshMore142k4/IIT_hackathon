# To generate a new secret key:
# import random, string
# SECRET_KEY= "".join([random.choice(string.printable) for _ in range(24)])
# print (SECRET_KEY)

SECRET_KEY = "f8p|Ij0c{_P/r|{nb>Y/FmSE"

# Location of client data
DATA_SERVER = './saved_models'
DATA_FILE='x_test.csv'

# Location of saved model
MODEL_SERVER ='./saved_models'
MODEL_FILE='lgbm_best_model.pickle'
EXPLAINER_FILE='lgbm_explainer.pickle'

# best threshold of saved model
THRESHOLD=0.542

# Alerts configuration
# Absolute score change required to trigger an alert (e.g., 0.05 = 5 percentage points)
ALERT_DELTA_ABS = 0.05

# Scheduler cadence (seconds) for background refresher
# Default: 1800 seconds (30 minutes)
REFRESH_INTERVAL_SECONDS = 1800

# Demo mode: add small random jitter so alerts can trigger with static data
demo = {
    'DEMO_MODE': True,
    'DEMO_JITTER': 0.02  # ~2 percentage points standard deviation
}