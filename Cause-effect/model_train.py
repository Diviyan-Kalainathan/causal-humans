from lib_lopez_paz import experiment_challenge as lp

if __name__=="__main__":

    trainpath = []
    trainfilename = []
    targetfilename = []
    modelPath = []

    modelPath = "lib_lopez_paz/"

    # train dataset from Kaggle
    trainpath.append("datacauseeffect/CEpairs/CEdata/")
    trainfilename.append("CEfinal_train_pairs")
    targetfilename.append("CEfinal_train_target")

    # SUP1 train dataset from Kaggle
    trainpath.append("datacauseeffect/CEpairs/SUP1/")
    trainfilename.append("CEdata_train_pairs")
    targetfilename.append("CEdata_train_target")

    # SUP2 train dataset from Kaggle
    trainpath.append("datacauseeffect/CEpairs/SUP2/")
    trainfilename.append("CEdata_train_pairs")
    targetfilename.append("CEdata_train_target")

    # SUP3 train dataset from Kaggle
    trainpath.append("datacauseeffect/CEpairs/SUP3/")
    trainfilename.append("CEdata_train_pairs")
    targetfilename.append("CEdata_train_target")

    # lp.train(trainpath,trainfilename, targetfilename, modelPath)
    lp.trainIndep(trainpath,trainfilename, targetfilename, modelPath,alreadyFeturized = True )

