
d = {'key1':1, 'key2':2}

def f(**kwargs):
    print(kwargs['key1'])

f(**d)
