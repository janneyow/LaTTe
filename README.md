# LaTTe: LAnguage Trajectory TransformEr

<video width="100%" controls>
  <source src="./docs/media/ICRA2023_LaTTe_low.mp4" type="video/mp4"/>
</video>


## setup
<sub>_tested on Ubuntu 20.04_</sup>

[install anaconda](https://docs.anaconda.com/anaconda/install/linux/)

Environment setup
- From `environment.yml`
  ```bash
  conda env create --name latte --file=environment.yml
  conda activate latte
  ```
- If above fails, you can manually install the dependencies:
  ```bash
  conda create --name latte python=3.8
  conda activate latte
  # install tensorflow (this also installs cuda toolkit 11.8)
  conda install tensorflow-gpu
  python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

  # install pytorch for cuda toolkit 11.8
  conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

  # Verify install
  python3 -c "import torch; print(torch.cuda.is_available())"

  # Install other packages
  pip install opencv-python-headless
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git
  pip install transformers
  pip install similaritymeasures 
  pip install dqrobotics
  conda install matplotlib scipy scikit-learn rospkg
  ```


Download models

```
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1HQNwHlQUOPMnbPE-3wKpIb6GMBz5eqDg?usp=sharing -O models/.
```
Download synthetic dataset  
```
gdown --folder https://drive.google.com/drive/folders/1_bhWWa9upUWwUs7ln8jaWG_bYxtxuOCt?usp=sharing -O data/.
```

Download image dataset(optional)
```
gdown --folder https://drive.google.com/drive/folders/1Pok_sU_cK3RXZEpMfJb6SQIcCUfBjJhh?usp=sharing -O image_data/.
```



## Other relevant files
overview of the project
[model_overview.ipynb](model_overview.ipynb)

model variations and ablasion studies
[Results.ipynb](Results.ipynb)

user study interface
[user_study.py](user_study.ipynb)

generate syntetic dataset
[src/data_generator_script.py](src/data_generator_script.py)


