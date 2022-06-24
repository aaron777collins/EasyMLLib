# Easy-ML Library
* This framework was created to make researching easier

## Setup
1. run `sk-learn\Scripts\activate`
1. run `pip install -r requirements.txt`

## Development
1. Add your data to a folder called `data` (create it if it doesn't exist)
1. Make a folder within the `user_scripts` folder.
    * Make sure to include ImportParentFilesScript.py and import it within your scripts. It will allow you to directly import the library files
1. When reading your data, make sure to include `data` as the parent folder
    * Your code may look like:
    ```python
    from os import path
    yourPath = path.join("data", "somefile.csv")
    ```

## Running
1. run the following commands:
 ```shell
sk-learn\Scripts\activate
python user_scripts\[Your folder]\[Desired File].py
```
Example: 
`python user_scripts\Distracted_Driving\ModelTrainer.py`