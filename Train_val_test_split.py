import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/path', type=str, help='path to the label file')
    parser.add_argument('--split', default=0.1, type=float, help='Train/Test split ratio')
    parser.add_argument('--folds', default=5, type=int, help='No of folds in K-folds')
    parser.add_argument('--transformed', default=0, type=int, help='transformed labels')
    args = parser.parse_args()

    # loading the annotation file :-
    df = pd.read_csv(args.dataset)
    df = df.sample(frac=1).reset_index(drop=True)

    # train-test split :- 
    X_train, X_test, Y_train, Y_test = train_test_split(df['path'], df['label'], test_size = args.split, stratify=df['label'], random_state=42)
    X_train = X_train.to_frame()
    X_test = X_test.to_frame()
    Y_train = Y_train.to_frame()
    Y_test = Y_test.to_frame() 
    joined_df = X_train.join(Y_train, how='inner')
    train_df = joined_df.reset_index(drop=True)
    joined_df = X_test.join(Y_test, how='inner')
    test_df = joined_df.reset_index(drop=True)
    
    
    if args.transformed == 1:
        os.makedirs(os.path.join(os.getcwd(), 'Transformed_Test'), exist_ok=True)
        test_df.to_csv(os.path.join(os.getcwd(), 'Transformed_Test', 'test.csv'), index=False)
    else:
        os.makedirs(os.path.join(os.getcwd(), 'Test'), exist_ok=True)
        test_df.to_csv(os.path.join(os.getcwd(), 'Test', 'test.csv'), index=False)
    


    # k-fold :- 
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    i=0
    if args.transformed == 1:
        os.makedirs(os.path.join(os.getcwd(), 'Transformed_Train_Val_split'), exist_ok=True)
    else:
        os.makedirs(os.path.join(os.getcwd(), 'Train_Val_split'), exist_ok=True)
        
    for train_idx, val_idx in kf.split(train_df['path'], train_df['label']):
        df_train = train_df.iloc[train_idx].reset_index(drop=True)
        train = str(i) + 'train.csv'
        if args.transformed == 1:
            df_train.to_csv(os.path.join(os.getcwd(), 'Transformed_Train_Val_split', train), index=False)
        else:    
            df_train.to_csv(os.path.join(os.getcwd(), 'Train_Val_split', train), index=False)

        df_val = train_df.iloc[val_idx].reset_index(drop=True)
        val = str(i) + 'val.csv'
        if args.transformed == 1:
            df_val.to_csv(os.path.join(os.getcwd(), 'Transformed_Train_Val_split', val), index=False)
        else:
            df_val.to_csv(os.path.join(os.getcwd(), 'Train_Val_split', val), index=False)
        

        i = i + 1




if __name__ == '__main__':
    main()