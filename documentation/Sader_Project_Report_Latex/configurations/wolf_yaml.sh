cluster_config:
    email: "default_wolf_email@email.com"

preProcessing:
    executable: "PreProcessing.py"
    missing_value: 0
    label_encoding: True

datasplit :
    executable : "Splitingdata.py"
    no_of_files : 1
    data_file : "diabetes.arff"
    output_folder : "output"
    parameters :
    - ["collection","list",-k,[3,5]]
    - ["collection","list",-r,[10,15,20]]

feature_extraction:
    algo1 :
        executable : "PCA.py"
        parameters :
            - ["single",-n,8]
            - ["single",-c,True]
            - ["single",-w,False]
    no_of_files : 1

feature_selection:
    algo1 :
        executable : "SVM_RFE.py"
        parameters :
            - ["single",-n,5]
            - ["single",-s,1]
    no_of_files : 1

algorithm :
    algo1 :
        executable : "RandomForest.py"
        parameters :
            - ["collection","range",-d,[5,10,1]]
            - ["collection","range",-t,[100,200,50]]
    algo2 :
        executable : "LinearDiscriminantAnalysis.py"
        parameters :
            - ["collection","list",-s,['svd', 'lsqr', 'eigen']]
            - ["single",-v,"auto"]
            - ["single",-t,0.0001]
    algo3 :
        executable: "DecisionTree.py"
    no_of_files : 1

metric_calculation :
    no_of_files : 1
    executable : "MetricCollection.py"
