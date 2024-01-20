# Titanic - Machine Learning from Disaster
---

<img src="https://m.media-amazon.com/images/I/811lT7khIrL._AC_UF894,1000_QL80_.jpg" style="max-width: 700px; height: 673px; margin: 0px; width: 471px;" alt="Titanic-photo">

## Introduction
This is my first Machine Learning Project. This repository contains my work for the Kaggle Titanic survival prediction competition. The challenge is to use machine learning to create a model that predicts which passengers survived the Titanic shipwreck based on a range of features.

## Portfolio Website with my BEST Deployed Models in the cloud
I also made a [custom interactive website for this project](https://michael-ye-titanic-home.streamlit.app/), which describes the project, my techniques, and most importantly, I deployed my best models and allowed users to pass in values which are given to the model in the cloud and gives back predictions of survival to the user. Try it out at https://michael-ye-titanic-home.streamlit.app/ 

## Project Description
The project employs a variety of machine learning models, including Random Forests, Gradient Boosted Trees, and Neural Networks, to predict survival. Techniques for data cleaning, feature engineering, and model tuning are thoroughly documented in the Jupyter notebooks.

## File Descriptions
- `titanic-neural-network (1).ipynb`: FULL MAIN jupyter notebook (including stuff from the CLEAN_titanic-RandomForest file) from start to finish for the Neural Network
- `CLEAN_titanic-neural-network.ipynb`: CLEAN means that I removed all unncessary data preprocessing stuff and only included the code and markdown cells which I ended up using in my final data preprocessing. CLEAN also includes the most recent neural network models
- `titanic-random-forest-classifier.ipynb`: FULL Jupyter notebook for the Random/Decision Forest Models (including all the stuff from the CLEAN_titanic-RandomForest)
- `CLEAN_titanic-RandomForest.ipynb`: CLEAN means that I removed all unncessary data preprocessing stuff and only included the code and markdown cells which I ended up using in my final data preprocessing. CLEAN also includes the most recent random/decision forest models
- `CLEAN_titanic-GradientBoostedTreesModel.ipynb`: Jupyter notebook for the Gradient Boosted Trees model.


- `model_plot.html`: Visualization of model performance.
- `model1_RF.png`: Image depicting the Random Forest model structure.
- `model3_GB_hyperparameters.json`: Hyperparameters used for the Gradient Boosted model.
- `model3_GB_summary.txt`: Summary of the Gradient Boosted model's performance.
- `model3_RF_hyperparameters.json`: Hyperparameters used for the Random Forest model.

## Installation
To set up this project, you will need Python 3.x and the following libraries: Pandas, NumPy, SciKit-Learn, Matplotlib, Seaborn, and TensorFlow. Installation can be done via `pip`:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

## Usage
Each model can be run by navigating to the respective Jupyter notebook and executing the cells in order. Ensure that you have Jupyter installed and run:
```bash
jupyter notebook
```

## Contributing
Contributions are welcome. Please open an issue first to discuss what you would like to change or add.

## Credits
Special thanks to [Kaggle](https://www.kaggle.com/) for hosting the dataset and challenge.

## License
This project is open source and available under the [MIT License](LICENSE.md).
