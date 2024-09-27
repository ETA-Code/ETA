# ETA
## what is ETA?
![Image file ](framework.png)
Explainable and Transferable Adversarial Attack for ML-Based Network Intrusion Detectors

ETA is a general Explainable Transfer-based black-box adversarial Attack framework aiming to 1) craft transferable adversarial examples across various types of ML models and 2) explain why adversarial examples and adversarial transferability exist in NIDSs.

## Implementation Notes
* Special Python dependencies: numpy,pandas,lightgbm,xgboost,catboost,pytorch,sklearn,matplotlib
* The source code has been tested with Python 3.6 on a MacOS 64bit machine

## Usage

1. **Set the root path**:
   For example, See line 2 in `test/test_evasion_binary_figure` :

   ``` python
    ROOT_PATH=''
    sys.path.append(ROOT_PATH)
    ```

2. **Set the datasets and models**:
   See line 59 in `eta/get_model_data.py` :

   ``` python
    parser.add('--algorithms', required=False, default=['lr', 'svm', 'mlp_keras','dt','xgboost','kitnet','diff-rf'])
    parser.add('--datasets',required=False, default='cicids2017'])
    ```

 3. **Start the interpretability**:
    * Source code in `eta/interpretability/fai`

    * Run
    ``` python
        python test/test_fai.py
    ```

 4. **Start the adversarial attack**:
    * Source code in `eta/attacks/evasion/zosgd_shap_sum.py`
    ``` python
        def ZOSGDShapSumMethod()
    ```
    * Run
    ``` python
        python test/test_evasion_binary_figure.py
    ```

