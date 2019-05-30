import os

data_dir = "C:\\Users\\mengnan.yi\\Desktop\\Study\\python_deep_learning\\jena_climate"
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

data = np.zeros((100,14))
for i in range(len(data)):
    data[i]=i
train = generator(data,0,50)
train = generator(data,0,30,lookback=10,batch_size=3)

x=next(train)
print('train_gen=',x)


def generator(data, min_index, max_index,lookback=20, delay=1,
              shuffle=False, batch_size=10, step=2):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
        
        