# ekstep-gender-classification
 Inference Modes 
 ```
 --model-path : set the path of the model
 --file-mode  : default False; set True for single file inference
 --file-path  : path to .wav file if --file-mode = True
 --csv-path   : path to csv containing multiple audio file paths
 --save-dir   : default Current directory; else give path to save predictions.csv 
 ```
 
 Single file inference
 ```
 python scripts/inference.py --model-path model/clf_svc.sav --file-mode True --file-path <filename>.wav

 ```
 
Csv mode inference
Create a csv containing multiple file paths
 ```
 python scripts/inference.py --model-path model/clf_svc.sav --csv-path <file_paths>.csv --save-dir <destination path>
 
 ```
