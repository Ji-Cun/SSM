import datetime
import pandas as pd
from src.SSM import accuracy

if __name__ == '__main__':
    UCR_43 = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'CricketX', 'CricketY',
              'CricketZ', 'DiatomSizeReduction', 'ECG200', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR',
              'FiftyWords', 'Fish', 'GunPoint', 'Haptics', 'InlineSkate', 'ItalyPowerDemand', 'Lightning2',
              'Lightning7', 'Mallat', 'MedicalImages', 'MoteStrain', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface1',
              'SonyAIBORobotSurface2', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'Trace',
              'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
              'Wafer', 'WordSynonyms', 'Yoga']


    df = pd.DataFrame(columns=['dataset', 'itr', 'feature number', 'acc'],
                      dtype=object)

    for itr in range(5):
        for featureNumber in range(50, 501, 50):
            for dataset in UCR_43:
                print('========================================================')
                print('dataset', dataset)
                print('itr', itr)
                print('feature number', featureNumber)
                print('time', datetime.datetime.now())
                acc,_ = accuracy(dataset=dataset, feature_number=featureNumber)

                df = df._append({'dataset': dataset, 'itr': itr, 'feature number': featureNumber, 'acc': acc},
                                    ignore_index=True)
                resultFileName = "..\\result\\ShapeletNumberTemp.csv"
                df.to_csv(resultFileName)
    resultFileName = "..\\result\\ShapeletNumberAccuracy.csv"
    df.to_csv(resultFileName)

