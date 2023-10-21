## Dependencies
- Matplotlib
- Numpy
- OpenCV (cv2)
- Python 
- Pillow (PIL)
- PyTorch
- scikit-learn
- torchvision

## File Directory
### For Dataset
```bash
AAI3001_SmallProj
├── EuroSAT_RGB
```

### Full Directory
```bash
AAI3001_SmallProj
├── EuroSAT_RGB
│   ├── AnnualCrop
│   ├── Forest
│   ├── HerbaceousVegetation
│   ├── Highway
│   ├── Industrial
│   ├── Pasture
│   ├── PermanentCrop
│   ├── Residential
│   ├── River
│   ├── SeaLake
├── Old Codes
│   ├── task1.py
│   ├── task2.py
├── AA13001 - Small Project.pdf
├── README.md
├── image_resnet50_model.pth
├── image_resnet50_multiLabel_model.pth
├── Task 1_Training and Validation Loss Curves.png
├── Task 2_Training and Validation Loss Curves.png
├── task1train.py
├── task1validate.py
├── task2train.py
├── task2validate.py
└── utils.py
```

## Python Scripts/Files
- utils.py 
    - holds the common functions for Tasks 1 and 2.
    - not required to run, the respective scripts for Tasks 1 and 2 will call this utils script.

### Task 1
- task1train.py
- task1validate.py

### Task 2
- task2train.py
- task2validate.py


## Code Execution
- To execute the Python scripts for the respective tasks, simply open your command prompt or terminal, navigate to the "<b>AAI3001_SmallProj</b>" directory and run the following commands:
    - <code>python task1train.py</code> to train the Task 1 model.
    - <code>python task1validate.py</code> to validate the Task 1 model.
    - <code>python task2train.py</code> to train the Task 2 model.
    - <code>python task2validate.py</code> to validate the Task 2 model.
