# Uncertainty-Quantification-in-DL-Models-for-Cervical-Cytology


Link to CRIC dataset :- https://database.cric.com.br/classification

split Train-Val-Test 
```
python Train_val_test_split.py --dataset='path to ur csv label file' 
```



Extract cells from the patches :-

<img width="631" alt="Screenshot 2024-04-10 at 5 00 39 PM" src="https://github.com/shubhamOjha1000/Uncertainty-Quantification-in-DL-Models-for-Cervical-Cytology/assets/72977734/617071f4-d97f-4904-84c1-acbf91e7adf8">


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

<img width="887" alt="Screenshot 2024-04-10 at 5 10 55 PM" src="https://github.com/shubhamOjha1000/Uncertainty-Quantification-in-DL-Models-for-Cervical-Cytology/assets/72977734/8d7a8fca-9af9-49a6-a101-351822b723cc">


<img width="786" alt="Screenshot 2024-04-10 at 5 11 58 PM" src="https://github.com/shubhamOjha1000/Uncertainty-Quantification-in-DL-Models-for-Cervical-Cytology/assets/72977734/f8b9312c-e8f8-4261-b97f-d6622ab1a854">

<img width="779" alt="Screenshot 2024-04-10 at 5 12 43 PM" src="https://github.com/shubhamOjha1000/Uncertainty-Quantification-in-DL-Models-for-Cervical-Cytology/assets/72977734/0298f7e4-b0a0-4e20-80ed-dc325877fc23">
