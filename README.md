# Pseq2Sites


Pseq2Sites is a sequence-based prediction model for binding site prediction. \
The model was trained with PDBbind and sc-PDB datasets, and tested with COACH420, HOLO4K, and CSAR benchmark datasets. 


## Requirements


python == 3.7 \
Pytorch == 1.7.1 \
Biopython == 1.79 \
RDKit == 2021.03.01 \
Openbabel == 2.4.1 \
Chimera == 1.16 \
[ProtTrans](https://github.com/agemagician/ProtTrans)

## Prepare features


### 1. Get sequence and binding site information
Pseq2Sites use protein sequence, so prepare protein sequence data to predict binding site. \
The code to extract protein sequence from 3D strcuture will be upload later.

Pseq2Sites predicts binding site for multi-chain proteins. \
Each chain sequence is separated by ',' delimiter.

If predict the binding site of single chain, separate each chain.

For more details input file format, see *_data.csv file in datasets dir.

### 2. Get amino acid-level embeddings
ProTrans is used to extract amino acid-level embeddings from protein sequences.

```
python gen_features.py -i ./datasets/COACH420_data.tsv -o ./datasets/COACH420_features.pkl -l True
```

When `-l` option is False, ignored binding site information (for test). \
For training, set option is True.

## Predict
To predict the binding site for prepared protein, use the following code.

```
python test.py -c configuraion.yml
```

The result of `test.py` consist of the following format:

```
PDB Pred_BS
protein1  pred_bs1
protein2  pred_bs2
protein3  pred_bs3
```

The code for performance evaluation will be uploaded later.

## Train 
Using training data, train Pseq2Sites with 5-fold CV.

```
python train.py -c configuration.yml
```

The trained models are saved in ./results/CV dir.
