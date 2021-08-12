# What's bird

In this project, our goal is to train an audio dataset and to predict the species of bird. In the dataset, there are five bird species, Bkcchi, blujay, bulori, chispa, wilfly. Each type has 100 records. 


The models we use include: ResNet50, EfficientNet_model.


Dataset is from: https://www.kaggle.com/c/birdsong-recognition/data

## Repository struct
```
├── notebooks          <- notebooks for data exploration, data augmentation and experiments of models.
├── scripts            <- Shell and command-line-friendly Python scripts 
├── requirement.txt    <- Requirement files to reproduce the analysis environment
└── README.md
```
## Set Up

### Using an isolated environment

1.It would be better to have a virtual enviroenment with all required pacakes installed. 
```
conda create --name env_name 
```

2. Activate environment with:

```
conda activate env_name
```

3. Install packages required for model train :
```
pip install -r requirements.txt
```
