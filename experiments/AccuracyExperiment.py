import datetime
import pandas as pd
from src.SPRP import accuracy

if __name__ == '__main__':
    UCR_43 = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'CricketX', 'CricketY',
              'CricketZ', 'DiatomSizeReduction', 'ECG200', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR',
              'FiftyWords', 'Fish', 'GunPoint', 'Haptics', 'InlineSkate', 'ItalyPowerDemand', 'Lightning2',
              'Lightning7', 'Mallat', 'MedicalImages', 'MoteStrain', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface1',
              'SonyAIBORobotSurface2', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'Trace',
              'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
              'Wafer', 'WordSynonyms', 'Yoga']

    shapeletNumber = {'Adiac': 150, 'Beef': 350, 'CBF': 150, 'ChlorineConcentration': 300, 'CinCECGTorso': 450,
                      'Coffee': 500, 'CricketX': 350, 'CricketY': 500, 'CricketZ': 400, 'DiatomSizeReduction': 200,
                      'ECG200': 400, 'ECGFiveDays': 150, 'FaceAll': 500, 'FaceFour': 100, 'FacesUCR': 250,
                      'FiftyWords': 256, 'Fish': 200, 'GunPoint': 150, 'Haptics': 400, 'InlineSkate': 450,
                      'ItalyPowerDemand': 350, 'Lightning2': 350, 'Lightning7': 450, 'Mallat': 200,
                      'MedicalImages': 400, 'MoteStrain': 350, 'OliveOil': 150, 'OSULeaf': 500,
                      'SonyAIBORobotSurface1': 100, 'SonyAIBORobotSurface2': 50, 'StarLightCurves': 350,
                      'SwedishLeaf': 400, 'Symbols': 100, 'SyntheticControl': 50, 'Trace': 50, 'TwoLeadECG': 100,
                      'TwoPatterns': 250, 'UWaveGestureLibraryX': 450, 'UWaveGestureLibraryY': 500,
                      'UWaveGestureLibraryZ': 450, 'Wafer': 100, 'WordsSynonyms': 150, 'Yoga': 450}

    df = pd.DataFrame(columns=['dataset', 'itr', 'feature number', 'acc'],
                      dtype=object)

    for dataset in UCR_43:
        featureNumber = shapeletNumber.get(dataset)
        for itr in range(5):
            print('========================================================')
            print('dataset', dataset)
            print('itr', itr)
            print('feature number', featureNumber)
            print('time', datetime.datetime.now())

            acc = accuracy(dataset=dataset, feature_number=featureNumber)
            
            df = df._append({'dataset': dataset, 'itr': itr, 'feature number': featureNumber, 'acc': acc},
                                    ignore_index=True)
            resultFileName = "..\\result\\AccuracyTemp.csv"
            df.to_csv(resultFileName)
    resultFileName = "..\\result\\Accuracy.csv"
    df.to_csv(resultFileName)

