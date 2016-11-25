#from lopez_paz import experiment_challenge as lp
from fonollosa import train as fo
import sys


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

    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        # # # Test dataset
        trainpairs.append("CauseEffectDataset/" + dataset + "_pairs.csv")
        traintargets.append("CauseEffectDataset/" + dataset+"_targets.csv")
        trainpublicinfo.append("CauseEffectDataset/" + dataset + "_publicinfo.csv")
        # featurizedData.append("lopez_paz/T_pairs_featurized.csv")
        #
    else: raise ValueError
    # # # A train dataset
    # trainpairs.append("CauseEffectDataset/A_pairs.csv")
    # traintargets.append("CauseEffectDataset/A_targets.csv")
    # trainpublicinfo.append("CauseEffectDataset/A_publicinfo.csv")
    # featurizedData.append("lopez_paz/A_pairs_featurized.csv")
    #
    # # # B train dataset
    # trainpairs.append("CauseEffectDataset/B_pairs.csv")
    # traintargets.append("CauseEffectDataset/B_targets.csv")
    # trainpublicinfo.append("CauseEffectDataset/B_publicinfo.csv")
    # featurizedData.append("lopez_paz/B_pairs_featurized.csv")

    # # C train dataset
    # trainpairs.append("CauseEffectDataset/C_pairs.csv")
    # traintargets.append("CauseEffectDataset/C_targets.csv")
    # trainpublicinfo.append("CauseEffectDataset/C_publicinfo.csv")
    # featurizedData.append("lopez_paz/C_pairs_featurized.csv")

    # # D train dataset
    # trainpairs.append("CauseEffectDataset/D_pairs.csv")
    # traintargets.append("CauseEffectDataset/D_targets.csv")
    # trainpublicinfo.append("CauseEffectDataset/D_publicinfo.csv")
    # featurizedData.append("lopez_paz/D_pairs_featurized.csv")
    #
    # # # E train dataset
    # trainpairs.append("CauseEffectDataset/E_pairs.csv")
    # traintargets.append("CauseEffectDataset/E_targets.csv")
    # trainpublicinfo.append("CauseEffectDataset/E_publicinfo.csv")
    # featurizedData.append("lopez_paz/E_pairs_featurized.csv")
    #
    # trainpairs.append("CauseEffectDataset/F_pairs.csv")
    # traintargets.append("CauseEffectDataset/F_targets.csv")
    # trainpublicinfo.append("CauseEffectDataset/F_publicinfo.csv")
    # featurizedData.append("lopez_paz/F_pairs_featurized.csv")

    # if(modelToTrain == 1):

    # lp.featurizeData(trainpairs, featurizedData)

    # modelpath = "lopez_paz/pickles/modelCDE_David"
    #
    #
    # lp.train(featurizedData,traintargets, modelpath)
    #
    # featurizedData = []
    # featurizedData.append("lopez_paz/C_pairs_featurized.csv")
    # featurizedData.append("lopez_paz/D_pairs_featurized.csv")
    # traintargets = []
    # traintargets.append("CauseEffectDataset/C_targets.csv")
    # traintargets.append("CauseEffectDataset/D_targets.csv")
    #
    # modelpath = "lopez_paz/pickles/modelCD_David"
    # lp.train(featurizedData,traintargets, modelpath)
    #
    # featurizedData = []
    # featurizedData.append("lopez_paz/C_pairs_featurized.csv")
    # featurizedData.append("lopez_paz/E_pairs_featurized.csv")
    # traintargets = []
    # traintargets.append("CauseEffectDataset/C_targets.csv")
    # traintargets.append("CauseEffectDataset/E_targets.csv")
    #
    #
    # modelpath = "lopez_paz/pickles/modelCE_David"
    # lp.train(featurizedData,traintargets, modelpath)
    #
    # featurizedData = []
    # featurizedData.append("lopez_paz/D_pairs_featurized.csv")
    # featurizedData.append("lopez_paz/E_pairs_featurized.csv")
    # traintargets = []
    # traintargets.append("CauseEffectDataset/D_targets.csv")
    # traintargets.append("CauseEffectDataset/E_targets.csv")
    #
    # modelpath = "lopez_paz/pickles/modelDE_David"
    # lp.train(featurizedData,traintargets, modelpath)


    # elif (modelToTrain == 2):

    featurizedData = []
    featurizedData.append("fonollosa/"+dataset+"_pairs_featurized.csv")
    # featurizedData.append("fonollosa/C_pairs_featurized.csv")
    # featurizedData.append("fonollosa/D_pairs_featurized.csv")
    # featurizedData.append("fonollosa/E_pairs_featurized.csv")
    # featurizedData.append("fonollosa/F_pairs_featurized.csv")

    traintargets = []
    # traintargets.append("CauseEffectDataset/T_targets.csv")
    # traintargets.append("CauseEffectDataset/C_targets.csv")
    # traintargets.append("CauseEffectDataset/D_targets.csv")
    # traintargets.append("CauseEffectDataset/E_targets.csv")
    # traintargets.append("CauseEffectDataset/F_targets.csv")

    modelpath = "fonollosa/modelCDE_Fonollosa.pkl"
    fo.train(trainpairs, trainpublicinfo, traintargets,featurizedData, modelpath)

    featurizedData = []
    featurizedData.append("fonollosa/C_pairs_featurized.csv")
    featurizedData.append("fonollosa/D_pairs_featurized.csv")
    traintargets = []
    traintargets.append("CauseEffectDataset/C_targets.csv")
    traintargets.append("CauseEffectDataset/D_targets.csv")

    modelpath = "fonollosa/modelCD_Fonollosa.pkl"
    fo.train(trainpairs, trainpublicinfo, traintargets, featurizedData, modelpath)

    featurizedData = []
    featurizedData.append("fonollosa/C_pairs_featurized.csv")
    featurizedData.append("fonollosa/E_pairs_featurized.csv")
    traintargets = []
    traintargets.append("CauseEffectDataset/C_targets.csv")
    traintargets.append("CauseEffectDataset/E_targets.csv")

    modelpath = "fonollosa/modelCE_Fonollosa.pkl"
    fo.train(trainpairs, trainpublicinfo, traintargets, featurizedData, modelpath)

    featurizedData = []
    featurizedData.append("fonollosa/D_pairs_featurized.csv")
    featurizedData.append("fonollosa/E_pairs_featurized.csv")
    traintargets = []
    traintargets.append("CauseEffectDataset/D_targets.csv")
    traintargets.append("CauseEffectDataset/E_targets.csv")

    modelpath = "fonollosa/modelDE_Fonollosa.pkl"
    fo.train(trainpairs, trainpublicinfo, traintargets, featurizedData, modelpath)
