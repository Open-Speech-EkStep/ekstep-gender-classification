# ekstep-gender-classification

This repository is a part of [Vakyansh's](https://open-speech-ekstep.github.io/) recipes to build state of the art Speech Recogniition Model.

To infer multiple audio files, place the file paths in a csv file and set the mode accordingly :
 
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
 
Csv mode inference </br>
Create a csv containing multiple file paths of the audios
 ```
 python scripts/inference.py --model-path model/clf_svc.sav --csv-path <file_paths>.csv --save-dir <destination path>
 
 ```

Inference outputs for csv and single inference modes:
```
0 : Male
1 : Female
```
