import datetime
import pandas as pd
from src.SPRP import diversity

if __name__ == '__main__':
    #dataset = 'Coffee'
    dataset ='ECG200'

    shapeletNumber = 8
    print('========================================================')
    print('dataset', dataset)
    print('feature number', shapeletNumber)
    print('time', datetime.datetime.now())

    diversity(dataset=dataset, feature_number=shapeletNumber)
            


