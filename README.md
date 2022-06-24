# Easy-ML Library
* This library was created to make researching easier

## Setup
1. In your git bash, within the repo you want, run `git submodule add https://github.com/aaron777collins/Easy_ML.git`
1. copy the sk-learn folder from within Easy_ML to your directory and run `sk-learn\Scripts\activate`
1. run `pip install -r Easy_ML\requirements.txt`

## Development
1. Import any files you need as shown below:
```python
from Easy_ML.logger import Logger
```

## Running
1. run the following commands:
 ```shell
sk-learn\Scripts\activate
python [Desired File].py
```
Example: 
`python ModelTrainer.py`