Installation and running steps:

Step 1 - Create a python virtual environment

Step 2 - Activate the virtual environment and install dependencies into the current active virtual environment:
- `pip install -r requirements.txt`

Step 3 - Run train.py with the desired arguments to train the model. For training the model with doing data augmentation just on minority class, Run train_positives.py instead of train.py.
- `python ./train.py --task_name=hate --load_clusters=0 --dl_word_embeddings=1 --augmentation_model=fasttext --augmentation_percentage=0.2 --wise_augmentation=1`

Parameters:
- task_name: Name of the downstream task. It can be offensive or hate.
- load_clusters: Load clusters if they have been previously trained and saved. It can be 0 or 1.
- dl_word_embeddings: Download pretrained word embeddings. To use fastText, it should be 1.
- augmentation_model: Augmentation method. It can be fasttext, gpt2 or spelling. 
- augmentation_percentage: The ratio of the words substitute in performing augmentation using word embedding or spelling mistakes augmenter. It can be any number between 0 and 1.
- wise_augmentation: If wise augmentation should be performed. It can be 0 or 1.
- cluster_num: Number of clusters in K-means clustering. The default value is 50.
- dataset: Datasets root location. Default is './data'.
- save_dir: Save directory location. Default is './results'.

Step 4 - Use the notebook file analyze.ipynb to test the saved trained model on test set
