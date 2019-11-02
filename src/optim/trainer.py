import os
import glob
import time
from time import localtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from networks import network
import optim.loss as loss_func
"""
DIVA model

"""

#########################################################
################### create network ######################

def create_net(net_name, d_dim, y_dim, z_dim, train: bool=False):   
    
    if net_name in ("zx_encode_net", "zd_encode_net", "zy_encode_net", "net0", "net1", "net2"):
        model = network.encoder_input()
    elif net_name in ("d_encoder_net", "net3"):
        model = network.encoder_dy(d_dim)
    elif net_name in ("y_encoder_net", "net4"): 
        model = network.encoder_dy(y_dim)
    elif net_name in ("recon_net", "net5"):
        if train:
            model = network.decoder_input(z_dim*3)
        else:
            model = network.decoder_input(z_dim)
    elif net_name in ("d_class_net", "net6"):
        model = network.decoder_d(z_dim, d_dim)
    elif net_name in ("y_class_net", "net7"):
        model = network.decoder_y(z_dim, y_dim)
    
    return model

#########################################################
#################### Set optimizer ######################

def set_optimizer(net_name, lr, weight_decay):
#optimizer = optim.Adam([param for name, param in model.state_dict().iteritems()
#                        if 'foo' in name], lr=args.lr)
    optimizer = optim.Adam(net_name.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

#########################################################
#################### Set scheduler ######################

def set_scheduler(optimizer, lr_milestones):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    return scheduler

#########################################################
################# Create onehot vector ##################

def create_onehot(bs, d_dim, input):
    onehot = torch.FloatTensor(bs, d_dim)
    onehot.zero_()
    # print(domains.shape)
    onehot.scatter_(1, input.reshape(bs,1), 1)
    return onehot

#########################################################
################## Training the model ###################

def train (train_loader, net_list, train_status, alpha_d, alpha_y, beta, betavar, d_dim, y_dim, z_dim,
           optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
           lr_milestones: tuple = (), batch_size: int = 64,
           weight_decay: float = 1e-6,n_jobs_dataloader: int = 0):

    # Creating model
    # model_list = ["zx_encode_net", "zd_encode_net", "zy_encode_net", "d_encoder_net", "y_encoder_net", "recon_net", "d_class_net", "y_class_net"]
    # net_list = []
    
    optimizer = []
    scheduler = []
    for i, net_name in enumerate(net_list):    
        #creating networks
        # net_list.append(create_net(net_name, d_dim, y_dim, z_dim, training=training))
        net_list[i] = net_list[i].cuda()

        
        if train_status:
            # Set optimizers (Adam optimizer for now)
            optimizer.append(set_optimizer(net_list[i], lr, weight_decay))
            # Set learning rate scheduler
            scheduler.append(set_scheduler(optimizer[i], lr_milestones))
            net_list[i].train()
        elif not train_status and i>2: # for creating optimizer in test cases
            # Set optimizers (Adam optimizer for now)
            optimizer.append(set_optimizer(net_list[i], lr, weight_decay))
            # Set learning rate scheduler
            scheduler.append(set_scheduler(optimizer[i-3], lr_milestones))#i-3 since the third 
            net_list[i].train()

        else:
            net_list[i].eval()        

    start_time = time.time()
    
    if betavar:
        beta = 0

    loss_list = []
    for epoch in range(n_epochs):

        for i in range (len(scheduler)):
            scheduler[i].step()        
        if epoch in lr_milestones:
            print('  LR scheduler: new learning rate is %g' % float(scheduler[0].get_lr()[0]))
        
        print("beta = %.2f" % beta)
        idx_label_domain_output_zdim = []
        loss_epoch = 0.0
        loss_epoch1 = 0.0
        loss_epoch2 = 0.0
        loss_epoch3 = 0.0
        n_batches = 0
        epoch_start_time = time.time()

        for data in train_loader:

            #data includes inputs,  label, index, domain
            inputs, labels, idx, domains = data
            
            bs = labels.shape[0]
            d_onehot = create_onehot(bs, d_dim, domains)
            y_onehot = create_onehot(bs, y_dim, labels)

            inputs = inputs.cuda()
            y_onehot = y_onehot.cuda()
            d_onehot = d_onehot.cuda()
            domains = domains.cuda()
            labels = labels.cuda()
            
            for i in range (len(optimizer)):
                optimizer[i].zero_grad()

            if train_status:
                xzx, mu_xzx, logvar_xzx = net_list[0](inputs)
                xzd, mu_xzd, logvar_xzd = net_list[1](inputs)
                xzy, mu_xzy, logvar_xzy = net_list[2](inputs)
                
                dzd, mu_dzd, logvar_dzd = net_list[3](d_onehot)
                yzy, mu_yzy, logvar_yzy = net_list[4](y_onehot)
                
                output_x = net_list[5](torch.cat((xzx, xzd, xzy), dim=1))
                output_d = net_list[6](xzd) ## also experiment d_class_net(dzd)
                output_y = net_list[7](xzy) ## also experiment y_class_net(yzy)

                BCE = loss_func.bce_loss(output_x, inputs)
                loss_d = F.nll_loss(output_d, domains)# try reduction = 'sum'
                loss_y = F.nll_loss(output_y, labels)
                
                KLD_zx = loss_func.kld_loss(mu_xzx, logvar_xzx, torch.zeros_like(mu_xzx), torch.zeros_like(logvar_xzx))
                KLD_zd = loss_func.kld_loss(mu_xzd, logvar_xzd, mu_dzd, logvar_dzd)
                KLD_zy = loss_func.kld_loss(mu_xzy, logvar_xzy, mu_yzy, logvar_yzy)

                if torch.isnan(KLD_zx) or torch.isnan(KLD_zd) or torch.isnan(KLD_zy):
                    param_nan = True
                    print("BCE="+str(BCE.item())+" loss_d="+str(loss_d.item())+" loss_y="+str(loss_y.item()))
                    print("KLD_zx="+str(KLD_zx.item())+" KLD_zd="+str(KLD_zd.item())+" KLD_zy="+str(KLD_zy.item()))
                    print("break batch loop")
                    break
                        
                kld_loss = KLD_zx + KLD_zd + KLD_zy
                total_loss = BCE + (beta * kld_loss) + (alpha_d*loss_d) + (alpha_y*loss_y)

                total_loss.backward()
                for i in range(len(optimizer)):
                    optimizer[i].step()
                
                loss_epoch += total_loss.item()
                n_batches += 1
                idx_label_domain_output_zdim += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            domains.cpu().data.numpy().tolist(),
                                            output_x.cpu().data.numpy().tolist(),
                                            output_d.cpu().data.numpy().tolist(),
                                            output_y.cpu().data.numpy().tolist(),
                                            xzx.cpu().data.numpy().tolist(),
                                            xzd.cpu().data.numpy().tolist(),
                                            xzy.cpu().data.numpy().tolist(),
                                            dzd.cpu().data.numpy().tolist(),
                                            yzy.cpu().data.numpy().tolist(),))                   

            else:
                xzx, mu_xzx, logvar_xzx = net_list[0](inputs)
                xzd, mu_xzd, logvar_xzd = net_list[1](inputs)
                xzy, mu_xzy, logvar_xzy = net_list[2](inputs)
                
                # dzd, mu_dzd, logvar_dzd = net_list[3](d_onehot)
                # yzy, mu_yzy, logvar_yzy = net_list[4](y_onehot)
                
                output_xx = net_list[3](xzx)
                output_dx = net_list[4](xzd) ## also experiment d_class_net(dzd)
                output_yx = net_list[5](xzy) ## also experiment y_class_net(yzy)

                BCE1 = loss_func.bce_loss(output_xx, inputs)
                BCE2 = loss_func.bce_loss(output_dx, inputs)
                BCE3 = loss_func.bce_loss(output_yx, inputs)

                BCE1.backward()
                BCE2.backward()
                BCE3.backward()
                for i in range(len(optimizer)):
                    optimizer[i].step()
                
                loss_epoch1 += BCE1.item()
                loss_epoch2 += BCE2.item()
                loss_epoch3 += BCE3.item()
                n_batches += 1
        
                idx_label_domain_output_zdim += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            domains.cpu().data.numpy().tolist(),
                                            output_xx.cpu().data.numpy().tolist(),
                                            output_dx.cpu().data.numpy().tolist(),
                                            output_yx.cpu().data.numpy().tolist(),
                                            xzx.cpu().data.numpy().tolist(),
                                            xzd.cpu().data.numpy().tolist(),
                                            xzy.cpu().data.numpy().tolist(),))
        if train_status:
            loss_list.append(loss_epoch/n_batches)
            if epoch>=120 and (loss_list[epoch-100]-loss_epoch/n_batches < 0.1):
                print("Training finished: No changes in loss for the last 100 epochs")
                break
        else:
            loss_list.append(loss_epoch1/n_batches)
            if epoch>=80 and (loss_list[epoch-50]-loss_epoch1/n_batches < 0.1):
                print("Training finished: No changes in loss for the last 50 epochs")
                break            
        if epoch < 100 and betavar:
            beta+=0.01

        if train_status:
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}  T: {:.2f}  Loss: {:.4f}   BCE:{:.4f}   KLD_zx:{:.4f}   KLD_zd:{:.4f}   KLD_zy:{:.4f}'
                        .format(epoch + 1, n_epochs, epoch_train_time, loss_epoch / n_batches, BCE.item(), KLD_zx.item(), KLD_zd.item(), KLD_zy.item()))
        else:
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}  T: {:.2f}  BCE_x: {:.4f}   BCE_d:{:.4f}   BCE_y:{:.4f}'
                        .format(epoch + 1, n_epochs, epoch_train_time, loss_epoch1 / n_batches, loss_epoch2 / n_batches, loss_epoch3 / n_batches))
    train_time = time.time() - start_time
    print('training time: %.3f' % train_time)
    print('Finished training.')

    return net_list, idx_label_domain_output_zdim

#########################################################
################## Testing the model ####################

def test (net_test_list, test_loader, d_dim, y_dim, z_dim, train=True):
    print('Testing autoencoder...')
    loss_d = 0.0
    correct_d = 0.0
    loss_y = 0.0
    correct_y = 0.0
    pred_d = 0.0
    pred_y = 0.0
    BCE = 0.0
    BCE1 = 0.0
    BCE2 = 0.0
    BCE3 = 0.0

    start_time = time.time()
    
    idx_label_domain_output_zdim = []

    for i, _ in enumerate(net_test_list):
            net_test_list[i] = net_test_list[i].cuda()
            net_test_list[i].eval()

        # if train:
        #     if i < 3 or i > 4:
        #         net_test_list[i] = net_test_list[i].cuda()
        #         net_test_list[i].eval()
        # else:
        #     net_test_list[i] = net_test_list[i].cuda()
        #     net_test_list[i].eval()

    with torch.no_grad():
        for data in test_loader:

            inputs, labels, idx, domains = data
            inputs = inputs.cuda()
            domains = domains.cuda()
            labels = labels.cuda()
            idx = idx.cuda()
            
            if train:

                xzx, mu_xzx, logvar_xzx = net_test_list[0](inputs)
                xzd, mu_xzd, logvar_xzd = net_test_list[1](inputs)
                xzy, mu_xzy, logvar_xzy = net_test_list[2](inputs)
                # yzy, mu_yzy, logvar_yzy = y_encoder_net(y_onehot)
                # dzd, mu_dzd, logvar_dzd = d_encoder_net(d_onehot)

                output_x = net_test_list[5](torch.cat((xzx, xzd, xzy), dim=1))
                output_d = net_test_list[6](xzd) ## also experiment d_class_net(dzd)
                output_y = net_test_list[7](xzy) ## also experiment y_class_net(yzy)
                
                BCE += loss_func.bce_loss(output_x, inputs, reduction='sum').item()
                # loss_d += F.nll_loss(output_d, domains, reduction='sum').item() # try reduction = 'sum'
                loss_y += F.nll_loss(output_y, labels, reduction='sum').item()

                pred_d = output_d.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct_d += pred_d.eq(domains.view_as(pred_d)).sum().item()
                pred_y = output_y.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct_y += pred_y.eq(labels.view_as(pred_y)).sum().item()

                idx_label_domain_output_zdim += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            pred_y.cpu().data.numpy().tolist(),
                                            domains.cpu().data.numpy().tolist(),
                                            pred_d.cpu().data.numpy().tolist(),
                                            output_x.cpu().data.numpy().tolist(),
                                            output_d.cpu().data.numpy().tolist(),
                                            output_y.cpu().data.numpy().tolist(),
                                            xzx.cpu().data.numpy().tolist(),
                                            xzd.cpu().data.numpy().tolist(),
                                            xzy.cpu().data.numpy().tolist(),))  
            else:
                xzx, mu_xzx, logvar_xzx = net_test_list[0](inputs)
                xzd, mu_xzd, logvar_xzd = net_test_list[1](inputs)
                xzy, mu_xzy, logvar_xzy = net_test_list[2](inputs)

                output_xx = net_test_list[3](xzx)
                output_dx = net_test_list[4](xzd) ## also experiment d_class_net(dzd)
                output_yx = net_test_list[5](xzy) ## also experiment y_class_net(yzy)
                
                BCE1 += loss_func.bce_loss(output_xx, inputs, reduction='sum').item()
                BCE2 += loss_func.bce_loss(output_dx, inputs, reduction='sum').item()
                BCE3 += loss_func.bce_loss(output_yx, inputs, reduction='sum').item()

                idx_label_domain_output_zdim += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            domains.cpu().data.numpy().tolist(),
                                            output_xx.cpu().data.numpy().tolist(),
                                            output_dx.cpu().data.numpy().tolist(),
                                            output_yx.cpu().data.numpy().tolist(),
                                            xzx.cpu().data.numpy().tolist(),
                                            xzd.cpu().data.numpy().tolist(),
                                            xzy.cpu().data.numpy().tolist(),))
    # loss_d /= len(test_loader.dataset)
    if train:
        loss_y /= len(test_loader.dataset)

        print("Testing domain classification")
        print('\nTest set: Average loss: {:.4f}, Domain Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss_d, correct_d, len(test_loader.dataset),
            100. * correct_d / len(test_loader.dataset)))
        print('{{"metric": "Eval - NLL Loss", "value": {}}}'.format(loss_d))
        print('{{"metric": "Eval - Accuracy", "value": {}}}'.format(
            100. * correct_d / len(test_loader.dataset))) 

        print("Testing label classification")
        print('\nTest set: Average loss: {:.4f}, Label Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss_y, correct_y, len(test_loader.dataset),
            100. * correct_y / len(test_loader.dataset)))
        print('{{"metric": "Eval - NLL Loss", "value": {}}}'.format(loss_y))
        print('{{"metric": "Eval - Accuracy", "value": {}}}'.format(
            100. * correct_y / len(test_loader.dataset))) 

    test_time = time.time() - start_time
    print('DIVA model testing time: %.3f' % test_time)
    print('Finished testing DIVA model. \n')

    return idx_label_domain_output_zdim

###################################################
################## Save model #####################

def save_model(models_list, net_list, path, name):
    net_dict_state = {}
    for i, model_name in enumerate(models_list):
        net_dict_state[model_name] = net_list[i].state_dict()#.state_dict()

    local_time = time.localtime(time.time())
    f_name = strftime("%H%M%S", localtime())
    torch.save(net_dict_state , path+name+f_name+".pt")

###################################################
################# Get latest file #################
def get_latest_file (file_name, path):
    """
    This method returns the most recent seved file
    inputs:
    name, path
    output:
    latest_file
    """  
    f_path = glob.glob(path)
    f_list = []

    for file in f_path:
        # print(file, model_name)
        if file_name in file:
            # print(file)
            f_list.append(file)
    # print(len(f_list))
    latest_file = max(f_list, key=os.path.getctime)
    print("The most recent saved model's file: ",latest_file)
    return latest_file

###################################################
################### Load model ####################

def load_model(path, file_name, d_dim, y_dim, z_dim, train):
    net_list = []
    net_path = get_latest_file(file_name, path)
    checkpoint = torch.load(net_path)
    print(len(checkpoint))
    for i, key in enumerate(checkpoint):
        # print(key)
        net = create_net(key, d_dim, y_dim, z_dim, train=False)
        net.load_state_dict(checkpoint[key])
        net_list.append(net)
        print(key+'\tloaded')#, checkpoint)
    return net_list
