# Cric (Cytopathalogy)

split Train-Val-Test 
```
python Train_val_test_split.py --dataset='path to ur csv label file' 
```


Extract cells from the patches :-
```
python Cropping.py --dataset='path to the cell centre coordinates csv file' --img_dir='path to the cric img patches directory' --cell_img_dir='path to cell img directory'

```




Run baseline model
```
python main.py --num_classes=2 --num_epochs=100 --img_dir='path to the img directory' --model='CustomResNet'   --batch_size=256
```

Run Ensemble based model 
```
python main.py --num_classes=2 --num_epochs=100 --img_dir='path to the img directory' --model='EnsembleModel_ResNet' --multi_head_loss='meta_individual_multi_head_loss'  --batch_size=256

```

Run TTA model
```
python main.py --num_classes=2 --num_epochs=100 --img_dir='path to the joined tensor' --model='TTA_model' --multi_head_loss='avg_across_all_heads_loss'  --batch_size=32 --num_workers=4
```

Run MC Dropout model
```
python main.py --num_classes=2 --num_epochs=100 --img_dir='path to the img directory' --model='MC_Dropout_model' --multi_head_loss='avg_across_all_heads_loss'  --batch_size=256
```


