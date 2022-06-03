Data and Data Prepareing Tool
=
  * [BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
    * Image should be placed in `data/BP4D/img` (or you change the dataset path in `config/BP4D_config.yaml`)
     * For example: `data/BP4D/img/F001/T1/2440.jpg`
    * Image path, AU label, AU Relation label (for egde feature supervision in our paper) and AU class weight (for the weighted asymmetric loss) lists are in  `data/BP4D/list/`
    *   ```text
        data/BP4D/list/
        ├── data/BP4D/list/BP4D_test_img_path_fold1.txt
        ├── data/BP4D/list/BP4D_test_img_path_fold2.txt
        ├── data/BP4D/list/BP4D_test_img_path_fold3.txt
        ├── data/BP4D/list/BP4D_test_label_fold1.txt
        ├── data/BP4D/list/BP4D_test_label_fold2.txt
        ├── data/BP4D/list/BP4D_test_label_fold3.txt
        ├── data/BP4D/list/BP4D_train_img_path_fold1.txt
        ├── data/BP4D/list/BP4D_train_img_path_fold2.txt
        ├── data/BP4D/list/BP4D_train_img_path_fold3.txt     
        ├── data/BP4D/list/BP4D_train_label_fold1.txt
        ├── data/BP4D/list/BP4D_train_label_fold2.txt
        ├── data/BP4D/list/BP4D_train_label_fold3.txt
        ├── data/BP4D/list/BP4D_train_AU_relation_fold1.txt
        ├── data/BP4D/list/BP4D_train_AU_relation_fold2.txt
        ├── data/BP4D/list/BP4D_train_AU_relation_fold3.txt
        ├── data/BP4D/list/BP4D_weight_fold1.txt
        └── data/BP4D/list/BP4D_weight_fold2.txt
        └── data/BP4D/list/BP4D_weight_fold3.txt
        ```
  * [DISFA](http://mohammadmahoor.com/disfa-contact-form/)
    * Image should be placed in `data/DISFA/img` (or you change the dataset path in `config/DISFA_config.yaml`)
     * For example: `data/DISFA/img/SN001/0.png`
    * Image path, AU label, AU Relation label (for egde feature supervision in our paper) and AU class weight (for the weighted asymmetric loss) lists are in  `data/DISFA/list/`
    *   ```text
        data/BP4D/list/
        ├── data/DISFA/list/DISFA_test_img_path_fold1.txt
        ├── data/DISFA/list/DISFA_test_img_path_fold2.txt
        ├── data/DISFA/list/DISFA_test_img_path_fold3.txt
        ├── data/DISFA/list/DISFA_test_label_fold1.txt
        ├── data/DISFA/list/DISFA_test_label_fold2.txt
        ├── data/DISFA/list/DISFA_test_label_fold3.txt
        ├── data/DISFA/list/DISFA_train_img_path_fold1.txt
        ├── data/DISFA/list/DISFA_train_img_path_fold2.txt
        ├── data/DISFA/list/DISFA_train_img_path_fold3.txt  
        ├── data/DISFA/list/DISFA_train_label_fold1.txt
        ├── data/DISFA/list/DISFA_train_label_fold2.txt
        ├── data/DISFA/list/DISFA_train_label_fold3.txt
        ├── data/DISFA/list/DISFA_train_AU_relation_fold1.txt
        ├── data/DISFA/list/DISFA_train_AU_relation_fold2.txt
        ├── data/DISFA/list/DISFA_train_AU_relation_fold3.txt
        ├── data/DISFA/list/DISFA_weight_fold1.txt
        ├── data/DISFA/list/DISFA_weight_fold2.txt
        ├── data/DISFA/list/DISFA_weight_fold3.txt
        ```

**Tools for prepareing data**

After getting the datasets ([BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and [DISFA](http://mohammadmahoor.com/disfa-contact-form/))
, you can use our data processing tools in  `tool/`.

        tool/
        ├── BP4D_image_label_process.py
        ├── DISFA_image_label_process.py
        ├── BP4D_calculate_AU_class_weights.py
        ├── DISFA_calculate_AU_class_weights.py
        ├── BP4D_deal_AU_relation.py
        ├── DISFA_deal_AU_relation.py

Before testing or training on BP4D, run:
```
cd tool/
python BP4D_image_label_process.py
python BP4D_calculate_AU_class_weights.py
python BP4D_deal_AU_relation.py
```

Before testing or training on DISFA, run:
```
cd tool/
python DISFA_image_label_process.py
python DISFA_calculate_AU_class_weights.py
python DISFA_deal_AU_relation.py
```

`BP4D_image_label_process.py` and `DISFA_image_label_process.py` are two files to get `img_path_fold*.txt` and `label_fold*.txt` from source data files.
For example,
- to get `BP4D_train/test_img_path_fold*.txt` and `BP4D_train/test_label_fold*.txt`, run:
```
python BP4D_image_label_process.py
```

`BP4D_calculate_AU_class_weights.py` and `DISFA_calculate_AU_class_weights.py` are two files to get `weight_fold*.txt` from `label_fold*.txt`.
For example,
- to get `BP4D_weight_fold*.txt`, run:
```
python BP4D_calculate_AU_class_weights.py
```

`BP4D_deal_AU_relation.py` and `DISFA_deal_AU_relation.py` are two files to get `train_AU_relation_fold*.txt` from `label_fold*.txt`.
For example,
- to get `DISFA_train_AU_relation_fold*.txt`, run:
```
python DISFA_deal_AU_relation.py
```
