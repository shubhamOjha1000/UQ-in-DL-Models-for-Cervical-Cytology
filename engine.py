import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CustomMultiHeadLoss, CombinedMultiHeadLoss, MetaMultiHeadLoss, MetaCombinedMultiHeadLoss, entropy, stochastic_dropout, sample_mean_uncertainty, min_max_normalization

def train(data_loader, model, model_name, optimizer, Loss, multi_head_loss, epsilon, no_of_heads):

    # put the model in train mode
    model.train()

    for data in data_loader:
        feature = data[0].float()
        label = data[1]

        # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # Move the tensor to the selected device (CPU or CUDA)
        feature = feature.to(device)
        label = label.to(device)


        # do the forward pass through the model
        if model_name == 'CustomResNet' or model_name == 'Custom_ViT':

            outputs = model(feature)

            # calculate loss
            if Loss == 'cross_entropy':
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, label)
            
            elif Loss == 'binary_cross_entropy':
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(outputs, label)

            # zero grad the optimizer
            optimizer.zero_grad()

            # calculate the gradient
            loss.backward()

            # update the weights
            optimizer.step()


        else:

            if multi_head_loss == 'avg_across_all_heads_loss':
                # Forward pass
                outputs, _ = model(feature)

                # Calculate loss
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs.mean(dim=1), label)  # Average the predictions of all heads

                # zero grad the optimizer
                optimizer.zero_grad()

                # calculate the gradient
                loss.backward()

                # update the weights
                optimizer.step()


            elif multi_head_loss == 'individual_multi_head_loss':
                # Forward pass
                _, List= model(feature)

                # Calculate loss
                criterion = nn.CrossEntropyLoss()
                head_losses = []

                # multi-head loss:-
                for head_prediction in List:
                    head_loss = criterion(head_prediction, label)
                    head_losses.append(head_loss)


                # zero grad the optimizer
                optimizer.zero_grad()

                # calculate the gradient for all heads 
                for loss in head_losses:
                    loss.backward(retain_graph=True)

                # update the weights
                optimizer.step()



            elif multi_head_loss == 'combined_multi_head_loss':
                # Forward pass
                _, List= model(feature)

                # Calculate loss
                criterion = nn.CrossEntropyLoss()
                head_losses = []

                # multi-head loss:-
                for head_prediction in List:
                    head_loss = criterion(head_prediction, label)
                    head_losses.append(head_loss)

                # Sum the losses from all heads
                total_loss = sum(head_losses)


                # zero grad the optimizer
                optimizer.zero_grad()

                # calculate the gradient
                total_loss.backward()

                # update the weights
                optimizer.step()


            elif multi_head_loss == 'meta_individual_multi_head_loss':
                # Forward pass
                _, List= model(feature)

                # Calculate loss
                criterion = nn.CrossEntropyLoss()
                head_losses = []

                # multi-head loss:-
                for head_prediction in List:
                    head_loss = criterion(head_prediction, label)
                    head_losses.append(head_loss)


                min_indices = torch.argmin(torch.stack(head_losses), dim=0)
                epsilon = 0.2
                delta = torch.ones((len(head_losses), 1),  dtype=torch.float32)*epsilon
                device = torch.device("cuda")
                delta = delta.to(device)
                delta[min_indices] = 1 - epsilon
                modified_losses = [delta[idx] * head_losses[idx] for idx in range(len(head_losses))]


                # zero grad the optimizer
                optimizer.zero_grad()

                # calculate the gradient for all heads 
                for loss in modified_losses:
                    loss.backward(retain_graph=True)

                # update the weights
                optimizer.step()



            elif multi_head_loss == 'meta_combined_multi_head_loss':
                # Forward pass
                _, List= model(feature)

                # Calculate loss
                criterion = nn.CrossEntropyLoss()
                head_losses = []

                # multi-head loss:-
                for head_prediction in List:
                    head_loss = criterion(head_prediction, label)
                    head_losses.append(head_loss)


                min_indices = torch.argmin(torch.stack(head_losses), dim=0)
                epsilon = 0.2
                delta = torch.ones((len(head_losses), 1),  dtype=torch.float32)*epsilon
                device = torch.device("cuda")
                delta = delta.to(device)
                delta[min_indices] = 1 - epsilon
                modified_losses = [delta[idx] * head_losses[idx] for idx in range(len(head_losses))]

                #Sum the losses from all heads
                total_loss = sum(modified_losses)

                # zero grad the optimizer
                optimizer.zero_grad()

                #calculate the gradient
                total_loss.backward()

                # update the weights
                optimizer.step()


            





def val(data_loader, model, model_name, Loss, no_of_heads, combine_results, threshold, uncertainty_metric, Q):
    entropy_List = []
    var_List = []
    sample_mean_List = []
    val_loss_list = []
    final_output = []
    final_label = []

    # put model in evaluation mode
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            feature = data[0].float()
            label = data[1]

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)

            # do the forward pass through the model
            if model_name == 'CustomResNet' or model_name == 'Custom_ViT':
                outputs = model(feature)

                # calculate val loss
                if Loss == 'cross_entropy':
                    criterion = nn.CrossEntropyLoss()
                    temp_val_loss = criterion(outputs, label)
                    val_loss_list.append(temp_val_loss)
                    softmax_values = F.softmax(outputs, dim=1)
                    outputs = torch.argmax(softmax_values, dim=1).int()
                    
                elif Loss == 'binary_cross_entropy':
                    criterion = nn.BCEWithLogitsLoss()
                    val_loss = criterion(outputs, label)
                    outputs = (outputs > 0.5).int()


                OUTPUTS = outputs.detach().cpu().tolist()
                

            else:

                # Forward pass
                outputs, _ = model(feature)

                # Calculate loss
                criterion = nn.CrossEntropyLoss()
                temp_val_loss = criterion(outputs.mean(dim=1), label)
                val_loss_list.append(temp_val_loss)
                _, o = outputs.mean(dim=1).max(1)
                OUTPUTS = o.detach().cpu().tolist()

                if uncertainty_metric == 'entropy':

                    head_predictions = outputs
                    for i in range(len(head_predictions)):
                        #print(f'head_predictions:-{head_predictions[i]}')
                        softmax_output = F.softmax(head_predictions[i], dim=1)
                        s = softmax_output[:,o[i]].detach().cpu().tolist()
                        e = entropy(s)
                        entropy_List.append(e)

                
                elif uncertainty_metric == 'variance':

                    head_predictions = outputs
                    for i in range(len(head_predictions)):
                        #print(f'head_predictions:-{head_predictions[i]}')
                        softmax_output = F.softmax(head_predictions[i], dim=1)
                        s = softmax_output[:,o[i]].detach().cpu().tolist()
                        var = (torch.var(torch.tensor(s))*1000).detach().cpu().tolist()
                        var_List.append(var)


                elif uncertainty_metric == 'sample_mean':

                    head_predictions = outputs
                    for i in range(len(head_predictions)):
                        #print(f'head_predictions:-{head_predictions[i]}')
                        softmax_output = F.softmax(head_predictions[i], dim=1)
                        s = softmax_output[:,o[i]].detach().cpu().tolist()
                        sample_mean = sample_mean_uncertainty(s)
                        sample_mean_List.append(sample_mean)



            

            final_output.extend(OUTPUTS)
            final_label.extend(label.detach().cpu().tolist())
        
        if model_name == 'CustomResNet' or model_name == 'Custom_ViT':
            uncertainty_metric_List = []
            return final_output, final_label,  sum(val_loss_list)/len(val_loss_list), uncertainty_metric_List


        

        if uncertainty_metric == 'entropy':
            uncertainty_metric_List = min_max_normalization(entropy_List)

        elif uncertainty_metric == 'variance':
            uncertainty_metric_List = min_max_normalization(var_List)

        elif uncertainty_metric == 'sample_mean':
            uncertainty_metric_List = min_max_normalization(sample_mean_List)
            
    
        #calculate the threshold:-
        if Q == 'Q1':
            mask = torch.tensor(uncertainty_metric_List) > threshold
            # find the indices in uncertainty list above threshold
            idx = torch.nonzero(mask).squeeze()
        else:
            mask = torch.tensor(uncertainty_metric_List) < threshold
            # find the indices in uncertainty list below threshold
            idx = torch.nonzero(mask).squeeze()
            

        # Indices of elements to remove from final_output and final_label
        final_output = torch.tensor(final_output)
        final_label = torch.tensor(final_label)
        indices_to_remove = idx

        correct_indices = torch.nonzero(torch.eq(final_output, final_label)).squeeze()
        incorrect_indices = torch.nonzero(torch.ne(final_output, final_label)).squeeze()

        uncertainty_metric_List = torch.tensor(uncertainty_metric_List)
        correct_uncertainty_metric_List = uncertainty_metric_List[correct_indices]
        incorrect_uncertainty_metric_List = uncertainty_metric_List[incorrect_indices]
        

        mask = torch.ones(final_output.shape, dtype=torch.bool)
        mask[indices_to_remove] = False

        filtered_final_output = final_output[mask].detach().cpu().tolist()
        filtered_final_label = final_label[mask].detach().cpu().tolist()


    return filtered_final_output, filtered_final_label, sum(val_loss_list)/len(val_loss_list), correct_uncertainty_metric_List.tolist(), incorrect_uncertainty_metric_List.tolist()
        






def test(data_loader, model, model_name, combine_results, threshold, uncertainty_metric, Q):
    entropy_List = []
    var_List = []
    sample_mean_List = []
    final_output = []
    final_label = []


    # put model in evaluation mode
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            feature = data[0].float()
            label = data[1]

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)


            # do the forward pass through the model
            if model_name == 'CustomResNet' or model_name == 'Custom_ViT':
                outputs = model(feature)
                softmax_values = F.softmax(outputs, dim=1)
                outputs = torch.argmax(softmax_values, dim=1).int()
                OUTPUTS = outputs.detach().cpu().tolist()
                


            else:
                outputs, _ = model(feature)

                _, o = outputs.mean(dim=1).max(1)
                OUTPUTS = o.detach().cpu().tolist()

                if uncertainty_metric == 'entropy':

                    head_predictions = outputs
                    for i in range(len(head_predictions)):
                        #print(f'head_predictions:-{head_predictions[i]}')
                        softmax_output = F.softmax(head_predictions[i], dim=1)
                        s = softmax_output[:,o[i]].detach().cpu().tolist()
                        e = entropy(s)
                        entropy_List.append(e)

                
                elif uncertainty_metric == 'variance':

                    head_predictions = outputs
                    for i in range(len(head_predictions)):
                        #print(f'head_predictions:-{head_predictions[i]}')
                        softmax_output = F.softmax(head_predictions[i], dim=1)
                        s = softmax_output[:,o[i]].detach().cpu().tolist()
                        var = (torch.var(torch.tensor(s))*1000).detach().cpu().tolist()
                        var_List.append(var)


                elif uncertainty_metric == 'sample_mean':

                    head_predictions = outputs
                    for i in range(len(head_predictions)):
                        #print(f'head_predictions:-{head_predictions[i]}')
                        softmax_output = F.softmax(head_predictions[i], dim=1)
                        s = softmax_output[:,o[i]].detach().cpu().tolist()
                        sample_mean = sample_mean_uncertainty(s)
                        sample_mean_List.append(sample_mean)


            final_output.extend(OUTPUTS)
            final_label.extend(label.detach().cpu().tolist())
        if model_name == 'CustomResNet' or model_name == 'Custom_ViT':
            uncertainty_metric_List = []
            return final_output, final_label, uncertainty_metric_List

        
        if uncertainty_metric == 'entropy':
            uncertainty_metric_List = min_max_normalization(entropy_List)

        elif uncertainty_metric == 'variance':
            uncertainty_metric_List = min_max_normalization(var_List)

        elif uncertainty_metric == 'sample_mean':
            uncertainty_metric_List = min_max_normalization(sample_mean_List)
            
    
        #calculate the threshold:-
        if Q == 'Q1':
            mask = torch.tensor(uncertainty_metric_List) > threshold
            # find the indices in uncertainty list above threshold
            idx = torch.nonzero(mask).squeeze()
        else:
            mask = torch.tensor(uncertainty_metric_List) < threshold
            # find the indices in uncertainty list below threshold
            idx = torch.nonzero(mask).squeeze()
            

        # Indices of elements to remove from final_output and final_label
        final_output = torch.tensor(final_output)
        final_label = torch.tensor(final_label)
        indices_to_remove = idx

        correct_indices = torch.nonzero(torch.eq(final_output, final_label)).squeeze()
        incorrect_indices = torch.nonzero(torch.ne(final_output, final_label)).squeeze()

        uncertainty_metric_List = torch.tensor(uncertainty_metric_List)
        correct_uncertainty_metric_List = uncertainty_metric_List[correct_indices]
        incorrect_uncertainty_metric_List = uncertainty_metric_List[incorrect_indices]

        mask = torch.ones(final_output.shape, dtype=torch.bool)
        mask[indices_to_remove] = False

        filtered_final_output = final_output[mask].detach().cpu().tolist()
        filtered_final_label = final_label[mask].detach().cpu().tolist()

    return filtered_final_output, filtered_final_label, correct_uncertainty_metric_List.tolist(), incorrect_uncertainty_metric_List.tolist()
    

        










        



        




                    

                


        

                

                
                












                


                    



            




        


                






                
            



       
        

        
