## Src Code

*main.py* serves as the entrance of the entire model.

*Preprocess.py* preprocess original data and generate dataset.

*Corpus.py* loads and formats dataset.

Core code of SLRC model is in *model/*. Note that SLRC is an abstract class. Other models starting with SLRC_* inherit from SLRC and implement different Collaborative Filtering (CF) methods. There are four abstract functions need to be implemented.

