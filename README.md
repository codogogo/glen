# GLEN 

GLEN is a tool for specializing distributional word vectors for lexical entailment (i.e., hyponym-hypernym relation). The tool/code accompanies the following research paper: 

Glavaš, G., & Vulić, I. (2019, July). Generalized Tuning of Distributional Word Vectors for Monolingual and Cross-Lingual Lexical Entailment. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 4824-4830).

## Code

You will find the code for: 

(1) training the GLEN model for LE specialization described in the paper and
(2) Predicting LE scores using the GLEN instance you previously trained

### Training GLEN model

You simply need to run the script glen_train.py: 

*python glen_train.py*

The relevant configuration for model training is set in the file config.py. 
The paths to serialized lexical constraints (i.e., train and dev data) and
a distributional vector space are pre-set to a relative path ("resources" 
subdirectory, see subsection on Resources below). 

The model (which you name with model_name in config) needs to be stored
stored in ./resources/model. There is already one pre-trained GLEN model available (see Resources section below).
 
### Predicting LE scores with GLEN

You need to run the script glen_predict.py which takes two arguments: 

python glen_predict.py [pairs_path] [pred_path]

"pairs_path" is the path to the file containing word pairs for which you want
to predict the LE score. Each word pair should be in a separate line and the
two words should be separated by a tabulator (i.e., TAB; "\t").  

"preds_path" is the path to which you want to store the predictions of LE
scores (for the pairs from pairs_path). The model will predict scores for all
word pairs for which both words are found in the distributional vocabulary
(fasttext English distributional space is provided in ./resources/distributional). 
The output file (preds_path) contains word pairs with associated predicted 
LE scores.

## Resources 

The following resources are not part of the GitHub repo due to file sizes, but need to be obtained in order to successfully run GLEN:

(1) the lexical constraints we used as training examples to learn the explicit specialization function for LE (needed if you're running the training script)
(2) the serialized distributional space (fasttext, EN) which you can specialize (needed if you're running either the training or prediction script)
(3) the pre-trained GLEN specialization model (needed if you're running the prediction script)

These resources are available at: 
https://drive.google.com/file/d/1d6re6aOuoTWvJ8M821G1bh3rNPLHv2yd/view?usp=sharing

Download and unzip the archive and you will find three directories: 

(1) constraints
(2) distributional
(3) model

Copy these three directories into the "resources" subdirectory found in this repository. 

## Credits

If you're using GLEN in your research, please cite the following paper (BibTex entry): 

```
@inproceedings{glavas-vulic-2019-generalized,
    title = "Generalized Tuning of Distributional Word Vectors for Monolingual and Cross-Lingual Lexical Entailment",
    author = "Glava{\v{s}}, Goran  and
      Vuli{\'c}, Ivan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1476",
    doi = "10.18653/v1/P19-1476",
    pages = "4824--4830"
}
```
