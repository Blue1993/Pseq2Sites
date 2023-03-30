# Pseq2Sites


Pseq2Sites is a sequence-based prediction model for binding site prediction. \
The model was trained with PDBbind and sc-PDB datasets, and tested with COACH420, HOLO4K, and CSAR benchmark datasets. 


## Requirements

python==3.7 \
Pytorch==1.7.1 \
Biopython==1.79 \
RDKit==2021.03.01 \
Openbabel==2.4.1 \
Chimera==1.16 \
[ProtTrans](https://github.com/agemagician/ProtTrans)


## Extract protein sequences and binding sites from 3D structures
For Pseq2Sites training, protein sequence and binding site information must be prepared. \
Refer to the following code for extracting the protein sequence and binding site from the protein 3D structure.

'Preprocessing for PDBbind.ipynb' \
'Preprocessing for scPDB.ipynb' \
'Preprocessing for COACH420.ipynb' \
'Preprocessing for HOLO4K.ipynb'

The files for all complexes of each dataset can be downloaded from the following link.

PDBbind: (http://www.pdbbind.org.cn/) \
scPDB: (http://bioinfo-pharma.u-strasbg.fr/scPDB/) \
COACH420, HOLO4K: (https://github.com/rdk/p2rank-datasets)

After downloading the dataset, change the path in the ipynb file to the downloaded dataset dir.

## Prepare features


### 1. Get sequence and binding site information
Pseq2Sites use protein sequence, so prepare protein sequence data to predict binding site.

Pseq2Sites predicts binding site for multi-chain proteins. \
Each chain sequence is separated by ',' delimiter.

If predicting the binding site of a single chain on a multi-chain protein, separate each chain and write the chain info in each row.

For more details input file format, see *_data.csv file in datasets/examples dir.

### 2. Get amino acid-level embeddings
ProTrans is used to extract amino acid-level embeddings from protein sequences.

```
python gen_features.py -i ./datasets/COACH420_data.tsv -o ./datasets/COACH420_features.pkl -l True
```

When `-l` option is `False`, ignored binding site information (for test). \
For training, set option is `True`.

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

when using the label option.

```
python test.py -c configuraion.yml -l True
```

The result of `test.py` consist of the following format:

```
PDB BS  Pred_BS
protein1  bs1  pred_bs1
protein2  bs2  pred_bs2
protein3  bs3  pred_bs3
```

## Train 
Using training data, train Pseq2Sites with 5-fold CV.

```
python train.py -c configuration.yml
```

The trained models are saved in ./results/CV dir.

## Evaluation
For evaluation of binding site prediction results, refer to 'Binding site prediction example.ipynb'.

