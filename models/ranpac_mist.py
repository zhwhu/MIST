import copy
import logging
import numpy as np
import os
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import utils.tao as TL
from utils.contrastive_learning import normalize, Supervised_NT_xent_pre,Supervised_NT_xent_uni,Supervised_NT_xent_n_with_fisher, get_similarity_matrix, Supervised_NT_xent_n
# from inc_net import ResNetCosineIncrementalNet,SimpleVitNet, CosineLinear
# from utils.linears import SimpleContinualLinear
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,SimpleVitNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy
epochs=20
bcb_scale = 0.01
lrate = 0.01 
milestones = [60,100,140]
lrate_decay = 0.1
batch_size = 1

T = 2
weight_decay = 5e-4
num_workers = 8
class BaseLearner(object):
    def __init__(self, args):
        # print("bcb_lr ", lrate*bcb_scale)
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments=[]
        self._network = None
        self.args = args
        self.topk = 5
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.task_acc = [[0.0 for _ in range(10)] for _ in range(10)]

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, rpfc=self.args['use_RP'])["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _compute_accuracy(self, model, loader, linear=None):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            if linear:
                with torch.no_grad():
                    features = model.convnet(inputs)
                    outputs = linear(features)['logits']
            else:
                with torch.no_grad():
                    outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.is_dil =False
        self.is_clr = True
        self._network = SimpleVitNet(args, True, is_clr=self.is_clr)
        self._batch_size= args["batch_size"]
        self._network.to(self._device)
        self.frozen_masks = {name: torch.zeros_like(param, dtype=torch.bool) for name, param in self._network.named_parameters()}
        self.overlap_mask = {name: torch.zeros_like(param, dtype=torch.int32) for name, param in self._network.named_parameters()}
        self.args=args
        self.spr = args['spr'] 
        self.mi_epochs = args['mi_epochs']
        self.select_rate = args['select_rate']
        self.drop_rate = 1-args['drop_rate']
        resize_scale = (0.3, 1.0)
        hflip = TL.HorizontalFlipLayer().cuda()
        color_gray = TL.RandomColorGrayLayer(p=0.25).cuda()
        resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[224, 224, 3]).cuda()
        self.simclr_aug = transform = torch.nn.Sequential(
        hflip,
        color_gray,  
        resize_crop, )
        if args['spr'] != "run":
            self.linear = None
    def after_task(self):
        self._known_classes = self._classes_seen_so_far

    def update_fc(self, embed_dim, nb_classes):
        if self.linear == None:
            self.linear = SimpleContinualLinear(embed_dim, nb_classes)
        else:
            self.linear.update(nb_classes)
        self.linear.to(self._device)
    def replace_fc_with_ca(self,trainloader, task_size):
        self._network = self._network.eval()
        crct_num = self.total_classnum
        if self.args['use_RP']:
            #these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.fc.use_RP=True
            if self.args['M']>0:
                self._network.fc.W_rand=self.W_rand
            else:
                self._network.fc.W_rand=None
        sampled_data = []
        sampled_label = []
        num_sampled_pcls = 256
        for c_id in range(crct_num):
            t_id = c_id//task_size
            decay = (t_id+1)/(self._cur_task+1)*0.1
            cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device)*(0.9+decay) # torch.from_numpy(self._class_means[c_id]).to(self._device)
            cls_cov = self._class_covs[c_id].to(self._device)
            
            m = MultivariateNormal(cls_mean.float(), cls_cov.float())

            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
            sampled_data.append(sampled_data_single)                
            sampled_label.extend([c_id]*num_sampled_pcls)
        sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
        sampled_label = torch.tensor(sampled_label).long().to(self._device)

        inputs = sampled_data
        targets= sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y=target2onehot(label_list,self.total_classnum)
        if self.args['use_RP']:
            #print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            if self.args['M']>0:
                Features_h=torch.nn.functional.relu(Features_f@ self._network.fc.W_rand.cpu())
            else:
                Features_h=Features_f
            self.Q=self.Q+Features_h.T @ Y 
            self.G=self.G+Features_h.T @ Features_h
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
            self._network.fc.weight.data=Wo[0:self._network.fc.weight.shape[0],:].to(device='cuda')

    def replace_fc(self,trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            #these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.rpfc.use_RP=True
            if self.args['M']>0:
                self._network.rpfc.W_rand=self.W_rand
            else:
                self._network.rpfc.W_rand=None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y=target2onehot(label_list,self.total_classnum)
        if self.args['use_RP']:
            #print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            if self.args['M']>0:
                Features_h=torch.nn.functional.relu(Features_f@ self._network.rpfc.W_rand.cpu())
            else:
                Features_h=Features_f
            self.Q=self.Q+Features_h.T @ Y 
            self.G=self.G+Features_h.T @ Features_h
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
            self._network.rpfc.weight.data=Wo[0:self._network.rpfc.weight.shape[0],:].to(device='cuda')
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index=(label_list==class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype=Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index]+=class_prototype.to(device='cuda') #for dil, we update all classes in all tasks
                else:
                    #original cosine similarity approach of Zhou et al (2023)
                    class_prototype=Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index]=class_prototype #for cil, only new classes get updated

    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
    
    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)
        if self.args['use_RP']:
            #temporarily remove RP weights
            del self._network.rpfc
            self._network.rpfc=None
            self._network.update_rpfc(self._classes_seen_so_far)
        self._network.update_fc(self._classes_seen_so_far) #creates a new head with a new number of classes (if CIL)
        if self.is_dil == False:
            logging.info("Starting CIL Task {}".format(self._cur_task+1))
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._classes_seen_so_far-1))
        self.class_increments.append([self._known_classes, self._classes_seen_so_far-1])
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
        train_dataset_for_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="test", )
        self.train_loader_for_CPs = DataLoader(train_dataset_for_CPs, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._classes_seen_so_far), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_CPs)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def freeze_backbone(self,is_first_session=False):
        # Freeze the parameters for ViT.
        if 'vit' in self.args['convnet_type']:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False
        else:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False

    def show_num_params(self,verbose=False):
        # show total parameters and trainable parameters
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

    def _train(self, train_loader, test_loader, train_loader_for_CPs):

        self._stage1_training(train_loader, test_loader)
        if self.args['use_RP'] and self._cur_task == 0:
            self.setup_RP()

        self.replace_fc(train_loader_for_CPs)
        self.show_num_params()
        
    def _stage1_training(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''

        base_params = self._network.convnet.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        
        head_scale = 1.
        
        base_params = {'params': base_params, 'lr': 0.0001, 'weight_decay': weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': 0.01, 'weight_decay': weight_decay}
        
        network_params = [base_params, base_fc_params]

        base_simclr_params = [p for p in self._network.simclr.parameters() if p.requires_grad==True]

        base_simclr_params = {'params': base_simclr_params, 'lr': 0.01, 'weight_decay': weight_decay}
        network_params.append(base_simclr_params)
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        if self.spr == 'mist':
            self._mist_run(train_loader, test_loader, optimizer, scheduler)
        elif self.spr == 'rand':
            self._run_rand(train_loader, test_loader, optimizer, scheduler)
        elif self.spr == 'l2':
            self._run_l2(train_loader, test_loader, optimizer, scheduler)
        elif self.spr == 'grad':
            self._run_grad(train_loader, test_loader, optimizer, scheduler)
        elif self.spr == 'cerun':
            self.ce_run(train_loader, test_loader, optimizer, scheduler)

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
            if 'fc.weight' in name or 'fc.sigma'  in name or 'bias' in name or 'norm' in name or 'simclr' in name:  # 特殊处理最后两层  or 'cls_token' in name or 'pos_embed' in name or 'patch_embed' in name
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
            threshold2 = torch.quantile(scores_flat, 1-0.0005)
            current_task_frozen_mask = (scores_flat >= threshold2).view(scores.size())
            self.frozen_masks[name] |= current_task_frozen_mask  
        for name, mask in masks.items():
            if 'fc.weight' in name or 'fc.sigma' in name or 'bias' in name or 'norm' in name or 'simclr' in name:
                continue
            self.overlap_mask[name] += mask  
    def setup_RP(self):
        self.initiated_G=False
        self._network.rpfc.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            self._network.rpfc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(device='cuda')) #num classes in task x M
            self._network.rpfc.reset_parameters()
            self._network.rpfc.W_rand=torch.randn(self._network.fc.in_features,M).to(device='cuda')
            self.W_rand=copy.deepcopy(self._network.rpfc.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            #no RP, only decorrelation
            M=self._network.rpfc.in_features #this M is L in the paper
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

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
    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
            if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
                ori_classes = self._class_means.shape[0]
                assert ori_classes==self._known_classes
                new_class_means = np.zeros((self._total_classes, self.feature_dim))
                new_class_means[:self._known_classes] = self._class_means
                self._class_means = new_class_means
                new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
                new_class_cov[:self._known_classes] = self._class_covs
                self._class_covs = new_class_cov
            elif not check_diff:
                self._class_means = np.zeros((self._total_classes, self.feature_dim))
                self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))


            if check_diff:
                for class_idx in range(0, self._known_classes):
                    data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                        mode='test', ret_data=True)
                    idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                    vectors, _ = self._extract_vectors(idx_loader)
                    class_mean = np.mean(vectors, axis=0)

                    class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)
                    if check_diff:
                        log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)).item())
                        logging.info(log_info)
                        np.save('task_{}_cls_{}_mean.npy'.format(self._cur_task, class_idx), class_mean)

            if oracle:
                for class_idx in range(0, self._known_classes):
                    data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                        mode='test', ret_data=True)
                    idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                    vectors, _ = self._extract_vectors(idx_loader)

                    class_mean = np.mean(vectors, axis=0)

                    class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-5
                    self._class_means[class_idx, :] = class_mean
                    self._class_covs[class_idx, ...] = class_cov            

            for class_idx in range(self._known_classes, self._total_classes):

                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)


                class_mean = np.mean(vectors, axis=0)

                class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-4
                if check_diff:
                    log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)).item())
                    logging.info(log_info)
                    np.save('task_{}_cls_{}_mean.npy'.format(self._cur_task, class_idx), class_mean)
                    np.save('task_{}_cls_{}_mean_beforetrain.npy'.format(self._cur_task, class_idx), self._class_means[class_idx, :])
                self._class_means[class_idx, :] = class_mean
                self._class_covs[class_idx, ...] = class_cov

    def ce_run(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        self.update_fc(768, self._classes_seen_so_far-self._known_classes)
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                features = self._network.convnet(inputs)
                logits = self.linear(features)['logits']
                cur_targets = torch.where(targets - self._known_classes >= 0,
                                        targets - self._known_classes, -100)

                
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader, linear=self.linear)
                test_acc = self._compute_accuracy(self._network, test_loader, linear=self.linear)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)
    def _run_l2(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        self.update_fc(768, self._classes_seen_so_far-self._known_classes)

        for epoch in range(1, run_epochs + 1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                features = self._network.convnet(inputs)
                logits = self.linear(features)['logits']
                cur_targets = torch.where(targets - self._known_classes >= 0,
                                        targets - self._known_classes, -100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()

                for param in self._network.parameters():
                    if param.requires_grad and param.grad is not None:
                        abs_param = param.data.abs().view(-1)
                        num_elements = abs_param.numel()
                        top_k = int(0.05 * num_elements)
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
                train_acc = self._compute_accuracy(self._network, train_loader, linear=self.linear)
                test_acc = self._compute_accuracy(self._network, test_loader, linear=self.linear)
                info = f'Task {self._cur_task}, Epoch {epoch}/{epochs} => Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.3f}, Test_accy {test_acc:.3f}'
            else:
                info = f'Task {self._cur_task}, Epoch {epoch}/{epochs} => Loss {losses / len(train_loader):.3f}'
            logging.info(info)
    def _run_rand(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        self.update_fc(768, self._classes_seen_so_far-self._known_classes)
        for epoch in range(1, run_epochs + 1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                features = self._network.convnet(inputs)
                logits = self.linear(features)['logits']
                cur_targets = torch.where(targets - self._known_classes >= 0,
                                        targets - self._known_classes, -100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()

                for param in self._network.parameters():
                    if param.grad is not None:
                        grad_flat = param.grad.view(-1)
                        numel = grad_flat.numel()
                        k = max(1, int(0.05 * numel))  

                        selected_indices = torch.randperm(numel, device=grad_flat.device)[:k]
                        mask = torch.zeros_like(grad_flat, dtype=torch.bool)
                        mask[selected_indices] = True
                        param.grad *= mask.view_as(param.grad)

                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch % 5 == 0:
                train_acc = self._compute_accuracy(self._network, train_loader, linear=self.linear)
                test_acc = self._compute_accuracy(self._network, test_loader, linear=self.linear)
                info = f'Task {self._cur_task}, Epoch {epoch}/{epochs} => Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.3f}, Test_accy {test_acc:.3f}'
            else:
                info = f'Task {self._cur_task}, Epoch {epoch}/{epochs} => Loss {losses / len(train_loader):.3f}'
            logging.info(info)
    def _run_grad(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        self.update_fc(768, self._classes_seen_so_far-self._known_classes)
        for epoch in range(1, run_epochs + 1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                features = self._network.convnet(inputs)
                logits = self.linear(features)['logits']
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

                k = int(0.05 * all_grads.numel())
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
                train_acc = self._compute_accuracy(self._network, train_loader, linear=self.linear)
                test_acc = self._compute_accuracy(self._network, test_loader, linear=self.linear)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)