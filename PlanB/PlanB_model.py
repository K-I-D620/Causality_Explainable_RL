import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.models as models # Caused Segmentation fault
import ipdb

class CoPhyNet(nn.Module):
  def __init__(self, num_objects=5):
    super().__init__()

    self.K = num_objects

    # AB
    H = 32
    # The 56 is linked to the last dim of the struct_obs_ab
    self.mlp_inter = nn.Sequential(nn.Linear(2*(56),H),
                                   nn.ReLU(),
                                   nn.Linear(H,H),
                                   nn.ReLU(),
                                   nn.Linear(H,H),
                                   nn.ReLU(),
                                   )
    D = H
    self.D = D
    # 56 from -1 dim of x and H is from -1 dim of E
    self.mlp_out = nn.Sequential(nn.Linear(56+H, H),
                                 nn.ReLU(),
                                 nn.Linear(H, H))

    # RNN
    self.rnn = nn.GRU(D,H, num_layers=1, batch_first=True)

    # Stability
    self.mlp_inter_stab = nn.Sequential(nn.Linear(2*(H+56),H),
                                        nn.ReLU(),
                                        nn.Linear(H,H),
                                        nn.ReLU(),
                                        nn.Linear(H,H),
                                        nn.ReLU(),
                                        )
    self.mlp_stab = nn.Sequential(nn.Linear(H*2 + 56, H),
                                  nn.ReLU(),
                                  nn.Linear(H, 1))

    # Next position
    self.mlp_inter_delta = nn.Sequential(nn.Linear(2*(H+56),H),
                                         nn.ReLU(),
                                         nn.Linear(H,H),
                                         nn.ReLU(),
                                         nn.Linear(H,H),
                                         nn.ReLU(),
                                         )
    self.mlp_gcn_delta = nn.Sequential(nn.Linear(H*2 + 56, H),
                                       nn.ReLU(),
                                       nn.Linear(H, H))
    self.rnn_delta = nn.GRU(H,H, num_layers=1, batch_first=True)
    self.fc_delta = nn.Linear(H, 56)


    # args
    self.iterative_stab = True


  def gcn_on_AB(self, struct_obs_ab):
    list_out = []
    K = struct_obs_ab.size(1) # Number of objects = n+m
    T = struct_obs_ab.size(0) # Number of timesteps
    for i in range(T):
      x = struct_obs_ab[i,:,:56] # shape (n+m, 28 + 28)
      # print("gcn_on_AB: x shape ", x.shape)

      # compute interactions : e_t+1 = f(o_t^1,o_t^2,r)
      x_1 = x.unsqueeze(0)  # (1, n+m, 56)
      x_1 = x_1.unsqueeze(1).repeat(1, K, 1, 1)  # (1, n+m, n+m, 56)
      # print("gcn_on_AB: x_1 shape ", x_1.shape)
      x_2 = x.unsqueeze(0)  # (1, n+m, 56)
      x_2 = x_2.unsqueeze(2).repeat(1, 1, K, 1)  # (1, n+m, n+m, 56)
      # print("gcn_on_AB: x_2 shape ", x_2.shape)
      x_12 = torch.cat([x_1, x_2], -1).float() # (1, n+m, n+m, 2*(56))
      E = self.mlp_inter(x_12) # E (1, n+m, n+m, H)
      E = torch.mean(E, 1) # (1,n+m,H) Used in place of aggreg_E to reduce dimension
      # print("gcn_on_AB: E shape ", E.shape)

      # next position : o_t+1^1 = f(o_t^1,e_t+1)
      x_next_pos = x.unsqueeze(0)
      out = self.mlp_out(torch.cat([x_next_pos, E], -1).float()) # cat([x_next_pos, E], -1) = (1, n+m, 56 + H)
      # print("gcn_on_AB next pos out shape: ", out.shape) # out (1, n+m, H)
      list_out.append(out) # list_out prob added a dim at 0

    out = torch.stack(list_out, 1) # (B,T,K,H) in our case B could change

    return out

  def rnn_on_AB_up(self, seq_o, object_type=None):
    if object_type is not None:
      T = seq_o.shape[1] # Number of timesteps
      object_type = object_type.unsqueeze(2).repeat(1,1,T,1)

    K = seq_o.size(2) # Number of objects
    list_out = []
    for k in range(K):
      x = seq_o[:,:,k] # same as seq_o[:,:,k,:]

      if object_type is not None:
        x = torch.cat([x, object_type[:,k]], -1)

      out, _ = self.rnn(x) # (B,T,H)
      list_out.append(out[:,-1]) # Only use latent rep of last layer

    out = torch.stack(list_out, 1) # (B,K,H)
    return out

  def pred_stab(self, confounders, pose_t):
    """
    Given a timestep - predict the stability per object
    :param confounders: (B,K,H)
    :param pose_t: (B,K,28+28)
    :return: stab=(B,K,1)
    """
    list_stab = []
    x = pose_t # (B,K,28+28)
    x = torch.cat([confounders, x], -1) # (B, K, 56+H)
    K = x.size(1)

    # compute interactions : e_t+1 = f(o_t^1,o_t^2,r)
    x_1 = x.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, K, 56+H)
    x_2 = x.unsqueeze(2).repeat(1, 1, K, 1)  # (B, K, K, 56+H)
    x_12 = torch.cat([x_1, x_2], -1).float() # (B, K, K, 2*(56+H))
    E = self.mlp_inter_stab(x_12) # B,K,K,H
    E = torch.mean(E, 1) # (B,K,H) Used in place of aggreg_E to reduce dimension
    #print("pred_stab E shape: ", E.shape)

    # stability
    stab = self.mlp_stab(torch.cat([x, E], -1).float()) # cat([x, E], -1) = (B, K, H+H+56)
    #print("pred_stab stab shape: ", stab.shape)

    return stab # (B,K,1)

  def pred_D(self, confounders, struct_obs_c, B, T=30):
    """

    :param stability: (B,K,1)
    :param confounders: (B, n+m, H) a.k.a (B,K,H)
    :param input: struct_obs_c= (1, n+m, 28+28) a.k.a (T,K,56)
    :return: out=(B,T-1,n+m,56) stability=(B,T-1,n+m,1)
    """
    list_pose = []
    list_stability = []
    pose = struct_obs_c # (1, n+m, 28 + 28)
    for i in range(B-1):
      pose = torch.cat([pose, struct_obs_c], 0)
    # pose (B, n+m, 28+28)
    # print("pred_D pose before T loop: ", pose.shape)
    K = pose.size(1) # K = n+m
    list_last_hidden = []
    for i in range(T):
      # Stability prediction
      if i == 0 or self.iterative_stab == 'true':
        stability = self.pred_stab(confounders, pose)
      list_stability.append(stability)

      # Cat
      x = torch.cat([pose, confounders], -1) #.detach() ??? # x (B, n+m, 56 + H)
      # print("pred_D: x shape ", x.shape)

      # compute interactions : e_t+1 = f(o_t^1,o_t^2,r)
      x_1 = x.unsqueeze(1).repeat(1, K, 1, 1)  # (B, n+m, n+m, 56+H)
      x_2 = x.unsqueeze(2).repeat(1, 1, K, 1)  # (B, n+m, n+m, 56+H)
      x_12 = torch.cat([x_1, x_2], -1).float() # (B, n+m, n+m, 2*(56+H))
      E = self.mlp_inter_delta(x_12) # (B, n+m, n+m, H)
      E = torch.mean(E, 1) # (B,n+m,H) Used in place of aggreg_E to reduce dimension
      # print("pred_D: E shape ", E.shape)

      # next position : o_t+1^1 = f(o_t^1,e_t+1) with RNN on top
      _in = self.mlp_gcn_delta(torch.cat([x, E], -1).float()) # cat([x, E], -1)= shape(B, n+m, 2*(H)+56)
      # _in (B, n+m, H)
      # print("pred_D: _in shape ", _in.shape)
      B = _in.size(0) # Breadth of network
      list_new_hidden = []
      for k in range(K):
        if i == 0:
          hidden, *_ = self.rnn_delta(_in[:,[k]]) # _in shape (B,1,H)
          # hidden (B, 1, H)
          # print("pred_D: hidden shape ", hidden.shape)
        else:
          hidden, *_ = self.rnn_delta(_in[:,[k]], list_last_hidden[k].reshape(1,B,-1)) # (B,1,H)
          # hidden (B, 1, H)
          # print("pred_D: hidden shape ", hidden.shape)
        list_new_hidden.append(hidden)
      list_last_hidden = list_new_hidden
      hidden = torch.cat(list_last_hidden, 1) # (B,K,H)

      delta = self.fc_delta(hidden) # delta (B,K,56)

      if self.training:
        alpha = 0.01
        delta = delta * (1 - torch.sigmoid(stability/alpha)) # .detach() ???
      else:
        delta = delta * (1-(stability > 0).float())
      pose = pose + delta

      list_pose.append(pose)

    pose = torch.stack(list_pose, 1) # (B,T-1,K,56)
    stability = torch.stack(list_stability, 1) # (B,T-1,K,1)

    return pose, stability

  def forward(self, struct_obs_ab, struct_obs_c):

    T = struct_obs_ab.size(0) - 1

    # Run a GCN on AB
    seq_o = self.gcn_on_AB(struct_obs_ab) # (B, T, n+m, H)
    print("forward gcn on ab seq_o shape: ", seq_o.shape)

    # Run a RNN on the outputs of GCN
    confounders = self.rnn_on_AB_up(seq_o) # (B,K,H)
    print("forward rnn on gcn confounders shape: ", confounders.shape)

    # Breadth of NN
    B = confounders.shape[0]
    print("forward B: ", B)
    # pred
    out, stability = self.pred_D(confounders, struct_obs_c, B, T=T)
    print("forward out shape: ", out.shape)
    # stability = (B,T-1,K,1)
    stability = stability.squeeze(-1)
    print("forward stability shape: ", stability.shape)

    return out, stability
