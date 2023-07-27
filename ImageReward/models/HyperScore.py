import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import models
import numpy as np

patches = 25

class HyperScore(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(self.device)
        self.model_hyper.train(False)
        self.model_hyper.load_state_dict((torch.load('./pretrained/koniq_pretrained.pkl')))   
        
        self.pil_to_tensor = lambda x: torchvision.transforms.ToTensor()(x).unsqueeze_(0)
        self.tensor_to_pil = lambda x: torchvision.transforms.ToPILImage()(x.squeeze_(0))
        
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225))])
            
        self.upsample = torchvision.transforms.Resize((224, 224))
        
    def score(self, prompt, image_path):
        
        if (type(image_path).__name__=='list'):
            _, rewards = self.inference_rank(prompt, image_path)
            return rewards
            
        # image encode
        pil_image = Image.open(image_path)
        image = self.pil_to_tensor(pil_image).to(self.device)
        image_size = torchvision.transforms.functional.get_image_size(image)
        if image_size[0] < 224 or image_size[1] < 224:
            image = self.upsample(image)
        
        # score
        pred_scores = []
        for i in range(patches):
            tile = self.transforms(image.clone().detach())
            paras = self.model_hyper(tile)  # 'paras' contains the network weights conveyed to target network

            # Building target network
            model_target = models.TargetNet(paras).cuda()
            for param in model_target.parameters():
                param.requires_grad = False

            # Quality prediction
            pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
            pred_scores.append(float(pred.item()))
        score = np.mean(pred_scores)
            
        return score

    def inference_rank(self, prompt, generations_list):
        
        scores = []
        for generations in generations_list:
            # image encode
            image_path = generations
            pil_image = Image.open(image_path)
            image = self.pil_to_tensor(pil_image).to(self.device)
            image_size = torchvision.transforms.functional.get_image_size(image)
            if image_size[0] < 224 or image_size[1] < 224:
                image = self.upsample(image)            
            
            # score
            pred_scores = []
            for i in range(patches):
                tile = self.transforms(image.clone().detach())
                paras = self.model_hyper(tile)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
                pred_scores.append(float(pred.item()))
            scores.append(np.mean(pred_scores))
            
        score_dict = {scores[i]: i+1 for i in range(len(scores))} 
        sorted_dict = {key: idx+1 for idx, (key, value) in enumerate(dict(sorted(score_dict.items(), reverse=True)).items())}
        #print(sorted_dict)
        indices_rewards = {sorted_dict[key]: key for key, value in score_dict.items()}
        
        #sorted_dict = dict(sorted(score_dict.items(), key=lambda x:x[1], reverse=True))
        
        rewards = [*indices_rewards.values()]
        indices = [*indices_rewards.keys()]
        
        print(indices, rewards)
        
        return indices, rewards
