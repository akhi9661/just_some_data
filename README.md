# Background
 ---
### Deriving wavelength dependent parameters using relative spectral response (RSRs) for Landsat 8 and Sentinel 2.

---

### Usage 

If you already have the necessary dependencies (mentioned in step 3) installed in your environment, start from step 4. 
The installation is much easier in a conda environment [miniconda or anaconda]

#### 1. Download miniconda/anaconda and install. 

- [miniconda](https://docs.conda.io/en/latest/miniconda.html)
 
#### 2. Open anaconda prompt and follow the steps mentioned below.

- Create a new environment named ENVNAME with the latest Python version:  
         
      conda create -n ENVNAME python
- Activate the new conda environment: 

      conda activate ENVNAME

#### 3. Install the necessary dependencies

The list of dependencies is available in `requirements.txt`. 

- Install package from conda-forge: 
      
      conda install -c conda-forge rasterio
      conda install -c conda-forge glob2
      conda install -c conda-forge pandas

Numpy, os, and IPython.display are installed by default.

#### 4. Open the link below and Ctrl + S to download the file as a python script.
- [Script](https://raw.githubusercontent.com/akhi9661/just_some_data/main/rsr.py)

#### 5. Run the script.
To run the script in anaconda prompt: 

    python rsr.py

#### 6. Follow the prompts.

----

Note: This is not a fully developed repository. The code is provided as it is and no help will be provided regarding the same.
