WOLF USER MANUAL:

This manual contains five parts:

I   WOLF Environment Setup

II  WOLF Installation 

III Run WOLF

IV  Results

V   Make Predictions



Part I: WOLF Environment Setup

1. Login to the ittc server
```
ssh login1.ittc.ku.edu
```
2. Add the following commands to your ~/.profile or run on your own each time
```
module load scikit-learn/0.18.0-Python-2.7.12
module load TensorFlow/1.11-cp27-gpu
```
3. Install eli5 on your profile (only need to do once)
```
module load Python/2.7.12
pip install --user eli5
```
4. Create a new directory for WOLF with desired name
```
mkdir DIRNAME
```
 

Part II: WOLF Installation 

1. Change to the directory where you want WOLF
```
cd DIRNAME
```
3. get the latest version of WOLF:
```
git clone https://github.com/CaseySader/WOLF_CL.git
```
 

Part III: Run WOLF

1. Change to the new WOLF directory
```
cd WOLF_CL
```
2. To run WOLF, some example config file, such as `wolf.yaml`, has been provided for testing. You may edit the following line:
   email: xxx@xx.xx.xx 
   in the culster_config section of `wolf.yaml` to change xxx@xx.xx.xx to your email so that you can get notifications of running status when you run WOLF using:
   ```
   python Wolf.py -i wolf.yaml.
   ```
 

Part IV: Results

The final results file generated from WOLF is stored in a file called `Results.xlsx`. An example file is provided in the WOLF/docs directory.
1. Get Results.xlsx
    (if off campus connect to kuanywhere.ku.edu through KU Anywhere VPN)
   
    1) Mac and Linux
      a. Open a terminal
      b. sftp to the ittc server ```sftp login1.ittc.ku.edu```
      c. Change to the WOLF directory ```cd DIRNAME/WOLF_CL```
      d. Download the results file ```get Results.xlsx```

    2) Windows
      You may use different kinds of SFTP clients to access login1.ittc.ku.edu. WinSCP is free SFTP client for windows, the details of WinSCP can be found in https://winscp.net/eng/index.php.      
2. Results.xlsx
   The content of `Results.xlsx` consists of metrics of different algorightms you specify in the WOLF configuration file.


Part V: Make Predictions
1. Once a model has been trained, it can be used to make predictions on a dataset in the same format as the training set. It can be run on its own or alongside another WOLF run.

2. WOLF is provided the dataset to predict on, the model to use, the name of the file to write predictions to (optional), the label to predict (must be provided if in the dataset), and a True/False value for calculating accuracy (can only be True if label provided)

3. Example of how to edit a yaml file for predictions can be seen in `wolf_predict.yaml` which can be run using
   ```
   python Wolf.py -i wolf_predict.yaml.
   ```
4. The output will be placed in the `predictions` folder
