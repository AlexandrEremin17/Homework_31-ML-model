import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
import dill

sched = BlockingScheduler(timezone=tzlocal.get_localzone())

df = pd.read_csv('../../Homework311/model/data/homework.csv')

with open('model/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)


@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(5)
    data['predicted_price_cat'] = model['model'].predict(data)
    data['price'] = df.price
    print(data[['id', 'price', 'predicted_price_cat']])


if __name__ == '__main__':
    sched.start()
