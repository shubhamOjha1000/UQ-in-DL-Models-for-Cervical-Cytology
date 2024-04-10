import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DeiTModel, DeiTConfig
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils import stochastic_dropout


class CustomResNet(nn.Module):
    def __init__(self, num_class):
        super(CustomResNet, self).__init__()
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Add a new fully connected layer 
        self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )

    def forward(self, x):
        # Forward pass through ResNet base
        x = self.resnet(x)
        # Flatten features
        x = torch.flatten(x, 1)
        # Final FC layer for classification (logits)
        x = self.fc(x)
        return x


class EnsembleModel_ResNet(nn.Module):
    def __init__(self, num_class, num_heads):
        super(EnsembleModel_ResNet, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads

        # Define ResNet50 model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        module_ResNet = list(resnet.children())[:-1]
        self.ResNet = nn.Sequential(*module_ResNet)

         # Create multiple heads for ResNet50
        self.heads_ResNet = nn.ModuleList()
        for _ in range(int(num_heads)):
            ResNet_fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads_ResNet.append(ResNet_fc)


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        List = []
        x1 = self.ResNet(x)
        x1 = torch.flatten(x1, 1)
        List_ResNet = []
        for head in self.heads_ResNet:
            output = head(x1)
            List_ResNet.append(output)
        List.extend(List_ResNet)
        head_ResNet = torch.cat(List_ResNet, dim=1)

        head_outputs = head_ResNet
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)

        return h, List










class Custom_DieT(nn.Module):
    def __init__(self, num_class):
        super(Custom_DieT, self).__init__()
        self.num_class = num_class

        # Define ViT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.ViT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False

        self.ViT_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )

    
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x2 = self.ViT(x).last_hidden_state[:, 0, :]
        x2 = self.ViT_fc(x2)
        return x2
    


class TTA_model(nn.Module):
    def __init__(self, num_class, num_samples):
        super(TTA_model, self).__init__()
        self.num_class = num_class
        self.num_samples = num_samples
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Add a new fully connected layer 
        self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )


    def forward(self, x):
        batch_size, _, _, _, _ = x.shape
        reshaped_tensor = x.permute(1, 0, 2, 3, 4)
        List = []
        
        for data in reshaped_tensor:
            device = torch.device("cuda")
            data = data.float()
            data = data.to(device)
            output = self.resnet(data)
            output = torch.flatten(output, 1)
            output = self.fc(output)
            List.append(output)
        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_samples, self.num_class)
        return h, List
    







"""

class TTA_model(nn.Module):
    def __init__(self, num_class, num_samples):
        super(TTA_model, self).__init__()
        self.num_class = num_class
        self.num_samples = num_samples
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Add a new fully connected layer 
        self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
        
    def get_tta_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=30),  # Random rotation up to 30 degrees
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Random perspective transformation
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Random affine transformation
            transforms.RandomGrayscale(p=0.1),  # Randomly convert image to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    
    def perform_tta(self, image_tensor):
        augmented_images = []
        for _ in range(self.num_samples):
            augmented_image = []
            for img in image_tensor:
                tta_transform = self.get_tta_transform()  # Get random transformation for each sample
                img_pil = transforms.ToPILImage()(img)  # Convert tensor to PIL Image
                augmented_img = tta_transform(img_pil)
                augmented_image.append(augmented_img)  
            augmented_images.append(torch.stack(augmented_image))
        return torch.stack(augmented_images)


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.perform_tta(x)
        List = []
        for data in x:
            print(data.shape)
            device = torch.device("cuda")
            output = self.resnet(data.to(device))
            output = torch.flatten(output, 1)
            output = self.fc(output)
            List.append(output)
            
        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_samples, self.num_class)
        return h, List
"""
    








class ResNet50_FE(nn.Module):
    def __init__(self):
        super(ResNet50_FE, self).__init__()
        # Load pre-trained ResNet50 model and freeze its weights
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer of ResNet50 with an identity layer
        self.resnet.fc = nn.Identity()
        # Dropout layer with dropout probability

    def forward(self, x):
        features = self.resnet(x)
        return features


class Dropout_model(nn.Module):
    def __init__(self, dropout_prob):
        super(Dropout_model, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        features = self.dropout(x)
        return features    
    

class MC_Dropout_model(nn.Module):
    def __init__(self, num_class, num_samples, dropout_prob):
        super(MC_Dropout_model, self).__init__()
        self.num_class = num_class
        self.num_samples = num_samples
        self.dropout_prob = dropout_prob
        self.FE_model = ResNet50_FE()
        self.Dropout_model = Dropout_model(dropout_prob)

        self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
        
        
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        
        features = self.FE_model(x)
        self.Dropout_model.train() 
        List = []
        for i in range(self.num_samples):
            output = self.Dropout_model(features)
            output = self.fc(output)
            List.append(output)
                
        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_samples, self.num_class)
        return h, List
    







"""

class ResNet50WithFC(nn.Module):
    def __init__(self, output_dim, dropout_prob):
        super(ResNet50WithFC, self).__init__()
        # Load pre-trained ResNet50 model and freeze its weights
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer of ResNet50 with an identity layer
        self.resnet.fc = nn.Identity()

        # Add a new fully connected layer
        self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, output_dim)
            )

        # Dropout layer with dropout probability
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        features = self.resnet(x)
        features = self.dropout(features)  # Apply dropout to the extracted features
        output = self.fc(features)
        return output


class MC_Dropout_model(nn.Module):
    def __init__(self, num_class, num_samples, dropout_prob):
        super(MC_Dropout_model, self).__init__()
        self.num_class = num_class
        self.num_samples = num_samples
        self.dropout_prob = dropout_prob
        self.model = ResNet50WithFC(num_class, dropout_prob)
        
    def forward(self, x):
        batch_size, _, _, _ = x.shape
         
        self.model.train() 
        List = []
        for i in range(self.num_samples):
            # with torch.no_grad():
                output = self.model(x)
                List.append(output)
                
        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_samples, self.num_class)
        return h, List
"""
    









class Multi_head_MLP(nn.Module):
    def __init__(self, num_class, num_heads):
        super(Multi_head_MLP, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
       
        # Create multiple heads
        self.heads = nn.ModuleList()

        for _ in range(num_heads):
            head = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads.append(head)


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        # Forward pass through ResNet base
        x = self.resnet(x)
        # Flatten features
        x = torch.flatten(x, 1)
        # Forward pass through multiple heads
        List = []
        for head in self.heads:
            output = head(x)
            List.append(output)
        
        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)
        return h, List



class DieT_Multi_head_MLP(nn.Module):
    def __init__(self, num_class, num_heads):
        super(DieT_Multi_head_MLP, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads

        # Define ViT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.ViT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False

        # Create multiple heads
        self.heads = nn.ModuleList()

        for _ in range(num_heads):
            head = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads.append(head)

    
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x2 = self.ViT(x).last_hidden_state[:, 0, :]
        # Forward pass through multiple heads
        List = []
        for head in self.heads:
            output = head(x2)
            List.append(output)
        
        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)
        
        return h, List
    


    






class Multi_head_CNN_MLP_1(nn.Module):
    def __init__(self, num_class, num_heads):
        super(Multi_head_CNN_MLP_1, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Create multiple heads
        self.head_CNN = nn.ModuleList()
        self.head_MLP = nn.ModuleList()

        for _ in range(num_heads):
            CNN = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
            self.head_CNN.append(CNN)

            MLP = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.head_MLP.append(MLP)

    
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        # Forward pass through ResNet base
        x = self.resnet(x)
        # Forward pass through multiple heads
        List = []
        for cnn_module, mlp_module in zip(self.head_CNN, self.head_MLP):
            output = cnn_module(x)
            output = torch.flatten(output, 1)
            output = mlp_module(output)
            List.append(output)

        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)
        return h, head_outputs
    








class Multi_head_CNN_MLP_2(nn.Module):
    def __init__(self, num_class, num_heads):
        super(Multi_head_CNN_MLP_2, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Create multiple heads
        
        self.head_CNN = nn.ModuleList()
        self.head_MLP = nn.ModuleList()

        for _ in range(num_heads):
            CNN = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
            self.head_CNN.append(CNN)

            MLP = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.head_MLP.append(MLP)


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        # Forward pass through ResNet base
        x = self.resnet(x)

        # Forward pass through multiple heads 
        List = []
        for cnn_module, mlp_module in zip(self.head_CNN, self.head_MLP):
            output = cnn_module(x)
            output = torch.flatten(output, 1)
            output = mlp_module(output)
            List.append(output)

        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)
        return h, head_outputs
    





class EnsembleModel_CNN_DieT_1(nn.Module):
    def __init__(self, num_class):
        super(EnsembleModel_CNN_DieT_1, self).__init__()
        self.num_class = num_class
        # Define ResNet50 model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.CNN = nn.Sequential(*modules)

        self.CNN_fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
        
        # Define ViT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.ViT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False

        self.ViT_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
        

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        
        x1 = self.CNN(x)
        x1 = torch.flatten(x1, 1)
        x1 = self.CNN_fc(x1)
        
        x2 = self.ViT(x).last_hidden_state[:, 0, :]
        x2 = self.ViT_fc(x2)

        List = [x1, x2]
        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, 2, self.num_class)
        
        return h, head_outputs
    






class EnsembleModel_CNN_DieT_2(nn.Module):
    def __init__(self, num_class, num_heads):
        super(EnsembleModel_CNN_DieT_2, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads
        # Define ResNet50 model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.CNN = nn.Sequential(*modules)
 
        
        # Define ViT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.ViT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False

        # Create multiple heads for CNN
        self.heads_CNN = nn.ModuleList()
        for _ in range(int(num_heads/2)):
            CNN_fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads_CNN.append(CNN_fc)

        
        # Create multiple heads for ViT
        self.heads_ViT = nn.ModuleList()
        for _ in range(int(num_heads/2)):
            ViT_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads_ViT.append(ViT_fc)
            
    
        

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        List = []
        x1 = self.CNN(x)
        x1 = torch.flatten(x1, 1)
        List_CNN = []
        for head_CNN in self.heads_CNN:
            output = head_CNN(x1)
            List_CNN.append(output)
        List.extend(List_CNN)

        head_CNN = torch.cat(List_CNN, dim=1)
            

        x2 = self.ViT(x).last_hidden_state[:, 0, :]
        List_ViT = []
        for head_ViT in self.heads_ViT:
            output = head_ViT(x2)
            List_ViT.append(output)
        List.extend(List_ViT)

        head_ViT = torch.cat(List_ViT, dim=1)

        head_outputs = torch.cat((head_CNN, head_ViT), dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)

        return h, List
    








class EnsembleModel_ResNet_DenseNet(nn.Module):
    def __init__(self, num_class, num_heads):
        super(EnsembleModel_ResNet_DenseNet, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads

        # Define ResNet50 model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        module_ResNet = list(resnet.children())[:-1]
        self.ResNet = nn.Sequential(*module_ResNet)

        # Define DenseNet121 model
        densenet = models.densenet121(pretrained=True)
        for param in densenet.parameters():
            param.requires_grad = False
        module_DenseNet = list(densenet.children())[:-1]
        self.densenet = nn.Sequential(*module_DenseNet)


         # Create multiple heads for ResNet50
        self.heads_ResNet = nn.ModuleList()
        for _ in range(int(num_heads/2)):
            ResNet_fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads_ResNet.append(ResNet_fc)


        # Create multiple heads for DenseNet
        self.heads_DenseNet = nn.ModuleList()
        for _ in range(int(num_heads/2)):
            DenseNet_fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads_DenseNet.append(DenseNet_fc)




    def forward(self, x):
        batch_size, _, _, _ = x.shape
        List = []
        x1 = self.ResNet(x)
        x1 = torch.flatten(x1, 1)
        List_ResNet = []
        for head in self.heads_ResNet:
            output = head(x1)
            List_ResNet.append(output)
        List.extend(List_ResNet)
        head_ResNet = torch.cat(List_ResNet, dim=1)


        x2 = self.densenet(x)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        x2 = torch.flatten(x2, 1)
        List_DenseNet = []
        for head in self.heads_DenseNet:
            output = head(x2)
            List_DenseNet.append(output)
        List.extend(List_DenseNet)
        head_DenseNet = torch.cat(List_DenseNet, dim=1)

        head_outputs = torch.cat((head_ResNet, head_DenseNet), dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)

        return h, List
    










class EnsembleModel_ViT_DieT(nn.Module):
    def __init__(self, num_class, num_heads):
        super(EnsembleModel_ViT_DieT, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads

        # Define ViT model
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        self.ViT = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False


        # Define DieT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.DieT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.DieT.parameters():
            param.requires_grad = False

        
        # Create multiple heads for ViT
        self.heads_ViT = nn.ModuleList()
        for _ in range(int(num_heads/2)):
            ViT_fc = nn.Sequential(
                nn.Linear(config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads_ViT.append(ViT_fc)
        

        

         # Create multiple heads for DieT
        self.heads_DieT = nn.ModuleList()
        for _ in range(int(num_heads/2)):
            DieT_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads_DieT.append(DieT_fc)

    

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        List = []

        x1 = self.ViT(x).last_hidden_state[:, 0, :]
        List_ViT = []
        for head in self.heads_ViT:
            output = head(x1)
            List_ViT.append(output)
        List.extend(List_ViT)
        head_ViT = torch.cat(List_ViT, dim=1)



        x2 = self.DieT(x).last_hidden_state[:, 0, :]
        List_DeiT = []
        for head in self.heads_DieT:
            output = head(x2)
            List_DeiT.append(output)
        List.extend(List_DeiT)
        head_DeiT = torch.cat(List_DeiT, dim=1)



        head_outputs = torch.cat((head_ViT, head_DeiT), dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)

        return h, List


    





        

















        

        
            