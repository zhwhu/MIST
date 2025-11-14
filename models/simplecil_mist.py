import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
import utils.tao as TL
from utils.contrastive_learning import normalize, Supervised_NT_xent_uni,Supervised_NT_xent_pre,Supervised_NT_xent_n_with_fisher, get_similarity_matrix, Supervised_NT_xent_n
import copy
drop_rate = 0.1
num_workers = 8
batch_size=32
miepochs = 20
epochs = 20
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True, is_clr=True).to(self._device)
        self.args=args
        self.spr = args['spr'] 
        self.mi_epochs = args['mi_epochs']
        self.select_rate = args['select_rate']
        self.drop_rate = 1-args['drop_rate']
        self.frozen_masks = {name: torch.zeros_like(param, dtype=torch.bool) for name, param in self._network.named_parameters()}
        self.overlap_mask = {name: torch.zeros_like(param, dtype=torch.int32) for name, param in self._network.named_parameters()}
        resize_scale = (0.3, 1.0)
        hflip = TL.HorizontalFlipLayer().cuda()
        color_gray = TL.RandomColorGrayLayer(p=0.25).cuda()
        resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[224, 224, 3]).cuda()
        self.simclr_aug = transform = torch.nn.Sequential(
        hflip,
        color_gray,  
        resize_crop, )
    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self,trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding=model.convnet(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list=np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            self._network.fc.weight.data[class_index]=proto
        return model

   
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._sparse_training(self.train_loader, self.test_loader)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        
        self._network.to(self._device)
        self.replace_fc(train_loader_for_protonet, self._network, None)


    def _sparse_training(self, train_loader, test_loader):
        lrate = 0.01 
        weight_decay = 5e-4
        milestones = [60,100,140]
        lrate_decay = 0.1
        base_params = self._network.convnet.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]

        base_params = {'params': base_params, 'lr': lrate*0.01, 'weight_decay': weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': lrate, 'weight_decay': weight_decay}
        
        network_params = [base_params, base_fc_params]
   
        base_simclr_params = [p for p in self._network.simclr.parameters() if p.requires_grad==True]
        base_simclr_params = {'params': base_simclr_params, 'lr': lrate, 'weight_decay': weight_decay}
        network_params.append(base_simclr_params)

        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        if self.spr == 'run':
            self._run(train_loader, test_loader, optimizer, scheduler)
        elif self.spr == 'rand':
            self._run_rand(train_loader, test_loader, optimizer, scheduler)
        elif self.spr == 'l2':
            self._run_l2(train_loader, test_loader, optimizer, scheduler)
        elif self.spr == 'grad':
            self._run_grad(train_loader, test_loader, optimizer, scheduler)
        elif self.spr == 'mist':
            self._mist_run(train_loader, test_loader, optimizer, scheduler)




        
    def _mist_run(self, train_loader, test_loader, optimizer, scheduler):
        forzen_mask_copy = copy.deepcopy(self.frozen_masks)
        response_scores = {}
        for epoch in range(1):
            response_accumulator = {name: torch.zeros_like(param) for name, param in self._network.named_parameters()}
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs_pair = torch.cat((inputs,self.simclr_aug(inputs)),dim=0)

                features, simclr, out = self._network(inputs_pair, is_clr = True)
                simclr = normalize(simclr)
                sim_matrix = 1 * get_similarity_matrix(simclr) 
                loss_cl = Supervised_NT_xent_n(sim_matrix,labels=targets,temperature=0.07)

                loss = loss_cl
                optimizer.zero_grad()
                loss.backward()

                for name, param in self._network.named_parameters():
                    if param.grad is not None:
                        param.grad[forzen_mask_copy[name]] = 0 
                        response_accumulator[name] += param.grad **2  / len(train_loader)
                losses += loss.item()

            scheduler.step()
        info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
            self._cur_task, epoch+1, self.mi_epochs, losses/len(train_loader))
        logging.info(info)

        # select updated params based on MI-Fisher
        response_scores = {name: acc / len(train_loader) for name, acc in response_accumulator.items()}
        masks = {}
        for name, scores in response_scores.items():
            if 'fc.weight' in name or 'fc.sigma'  in name or 'bias' in name or 'norm' in name or 'simclr' in name:  # 
                masks[name] = torch.ones_like(scores, dtype=torch.bool)
            else:
                scores_flat = scores.view(-1)
                threshold1 = torch.quantile(scores_flat, (1-self.select_rate))  
                masks[name] = (scores_flat >= threshold1).view(scores.size()) 

        for epoch in range(2, self.mi_epochs+1):
            self._network.train()
            losses = 0.
            milosses = 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs_pair = torch.cat((inputs,self.simclr_aug(inputs)),dim=0)

                features, simclr, out = self._network(inputs_pair, is_clr = True)
                simclr = normalize(simclr)
                sim_matrix = 1 * get_similarity_matrix(simclr) 
                loss_cl = Supervised_NT_xent_n(sim_matrix,labels=targets,temperature=0.07)

                loss = loss_cl

                optimizer.zero_grad()
                loss.backward()

                for name, param in self._network.named_parameters():
                    if param.grad is not None:
                        if 'fc.weight' not in name and 'fc.sigma' not in name and 'norm' not in name and 'simclr' not in name:
                            param.grad[forzen_mask_copy[name]] = 0 # 
                self.masked_update(dict(self._network.named_parameters()), masks, forzen_mask_copy, self.overlap_mask)
                optimizer.step()
                milosses += loss_cl.item()

            scheduler.step()

            info = 'Task {}, Epoch {}/{} => miLoss {:.3f}'.format(
                self._cur_task, epoch, self.mi_epochs,  milosses/len(train_loader))
            logging.info(info)
        for name, scores in response_scores.items():
            if 'fc.weight' in name or 'fc.sigma' in name or 'bias' in name or 'norm' in name or 'simclr' in name:
                continue
            scores_flat = scores.view(-1)
            threshold2 = torch.quantile(scores_flat, 1)
            current_task_frozen_mask = (scores_flat >= threshold2).view(scores.size())
            self.frozen_masks[name] |= current_task_frozen_mask  
        for name, mask in masks.items():
            if 'fc.weight' in name or 'fc.sigma' in name or 'bias' in name or 'norm' in name or 'simclr' in name:
                continue
            self.overlap_mask[name] += mask  

    def print_mask_coverage(self, masks):
        info = "Mask Coverage:"
        logging.info(info)
        for name, mask in masks.items():
            selected_count = mask.sum().item()
            total_count = mask.numel()
            coverage = selected_count / total_count * 100  # 
            info = f"  Layer {name}: Selected {selected_count}/{total_count} ({coverage:.2f}%)"
            logging.info(info)
    def update_percent(self, masks):
        info = "update_percent:"
        logging.info(info)
        all_count = 0
        mask_count = 0
        for name, mask in masks.items():
            selected_count = mask.sum().item()
            total_count = mask.numel()
            all_count += total_count
            mask_count += selected_count

        coverage = mask_count / all_count * 100  # 
        info = f"  Total: Selected {mask_count}/{all_count} ({coverage:.2f}%)"
        logging.info(info)

    def masked_update(self, params, masks, forzen_mask_copy, overlap_mask):
        for name, param in params.items():
            if param.grad is not None:
                random_mask = torch.zeros_like(param.grad)
                #randomly dropout
                random_mask[masks[name] == 1] = torch.bernoulli(torch.full_like(param.grad[masks[name] == 1], self.drop_rate))
                final_mask = masks[name] * random_mask  
                param.grad *= final_mask 
                
                if overlap_mask is not None and name in overlap_mask:
                    update_count = overlap_mask[name] 
                    penalty_factors = 0.9 ** update_count 
                    param.grad *= penalty_factors
                if 'fc.weight' not in name and 'fc.sigma' not in name and 'bias' not in name and 'norm' not in name:
                    param.grad[forzen_mask_copy[name]] = 0

    def unfreeze(self):
        for n, p in self._network.convnet.named_parameters():
            if 'fc' not in n:
                p.requires_grad = True
    def freeze(self):
        for n, p in self._network.convnet.named_parameters():
            if 'fc' not in n:
                p.requires_grad = False
    

     
    def _run_l2(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        for epoch in range(1, run_epochs + 1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)['logits']
                cur_targets = torch.where(targets - self._known_classes >= 0,
                                        targets - self._known_classes, -100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()

                for param in self._network.parameters():
                    if param.requires_grad and param.grad is not None:
                        abs_param = param.data.abs().view(-1)
                        num_elements = abs_param.numel()
                        top_k = int(0.005 * num_elements)
                        
                        if top_k == 0:
                            param.grad.zero_()
                            continue
                        
                        threshold = torch.topk(abs_param, top_k, largest=True)[0][-1]
                        mask = (param.data.abs() >= threshold).float()
                        param.grad *= mask.view_as(param.data)

                optimizer.step()
                losses += loss.item()

            scheduler.step()

            # === Logging ===
            if epoch % 5 == 0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = f'Task {self._cur_task}, Epoch {epoch}/{epochs} => Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.3f}, Test_accy {test_acc:.3f}'
            else:
                info = f'Task {self._cur_task}, Epoch {epoch}/{epochs} => Loss {losses / len(train_loader):.3f}'
            logging.info(info)
    def _run_rand(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        for epoch in range(1, run_epochs + 1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)['logits']
                cur_targets = torch.where(targets - self._known_classes >= 0,
                                        targets - self._known_classes, -100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()

                for param in self._network.parameters():
                    if param.grad is not None:
                        grad_flat = param.grad.view(-1)
                        numel = grad_flat.numel()
                        k = max(1, int(0.05 * numel))  # 

                        selected_indices = torch.randperm(numel, device=grad_flat.device)[:k]
                        mask = torch.zeros_like(grad_flat, dtype=torch.bool)
                        mask[selected_indices] = True
                        param.grad *= mask.view_as(param.grad)

                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch % 5 == 0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = f'Task {self._cur_task}, Epoch {epoch}/{epochs} => Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.3f}, Test_accy {test_acc:.3f}'
            else:
                info = f'Task {self._cur_task}, Epoch {epoch}/{epochs} => Loss {losses / len(train_loader):.3f}'
            logging.info(info)
    def _run_grad(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        for epoch in range(1, run_epochs + 1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)['logits']
                cur_targets = torch.where(targets - self._known_classes >= 0,
                                        targets - self._known_classes, -100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()

                grads = []
                for param in self._network.parameters():
                    if param.grad is not None:
                        grads.append(param.grad.detach().view(-1).abs())
                all_grads = torch.cat(grads)

                k = int(0.005 * all_grads.numel())
                if k == 0:
                    threshold = all_grads.max() + 1  #
                else:
                    threshold = torch.topk(all_grads, k, largest=True)[0][-1]

                for param in self._network.parameters():
                    if param.grad is not None:
                        mask = param.grad.abs() >= threshold
                        param.grad *= mask

                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)
    def _run(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)['logits']
                cur_targets = torch.where(targets-self._known_classes>=0,targets-self._known_classes,-100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)
