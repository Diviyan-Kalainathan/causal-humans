from lopez_paz import experiment_challenge as lp
from fonollosa import train as fo

if __name__=="__main__":

    """Model to train
    # 1 : Lopez-paz 2015
    # 2 : Fonollosa
    """

    modelToTrain = 2

    trainpairs = []
    traintargets = []
    trainpublicinfo = []

    featurizedData = []

    # # A train dataset
    trainpairs.append("CauseEffectDataset/A_pairs.csv")
    traintargets.append("CauseEffectDataset/A_targets.csv")
    trainpublicinfo.append("CauseEffectDataset/A_publicinfo.csv")

    featurizedData.append("lopez_paz/A_pairs_featurized.csv")

    # # B train dataset
    trainpairs.append("CauseEffectDataset/B_pairs.csv")
    traintargets.append("CauseEffectDataset/B_targets.csv")
    trainpublicinfo.append("CauseEffectDataset/B_publicinfo.csv")

    featurizedData.append("lopez_paz/B_pairs_featurized.csv")



    maxproc = 2

    if(modelToTrain == 1):

        modelpath = "lopez_paz/pickles/modelAB_David"

        lp.featurizeData(trainpairs, featurizedData, maxproc)
        lp.train(featurizedData,traintargets, modelpath)

    elif (modelToTrain == 2):

        modelpath = "fonollosa/modelAB_Fonollosa.pkl"
        fo.train(trainpairs, trainpublicinfo, traintargets, modelpath)


