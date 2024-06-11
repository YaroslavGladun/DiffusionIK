import torch
import matplotlib.pyplot as plt

from seed_epsilon_ik_model import SeedEpsilonIKModel
from seed_epsilon_ik_config import SeedEpsilonIKConfig
from seed_epsilon_ik_dataset import SeedEpsilonIKDataset
from affine_loss import AffineLoss
from fk import FK

config = SeedEpsilonIKConfig()
model = SeedEpsilonIKModel(torch.device('cuda'), config).to(torch.device('cuda'))
model.load_state_dict(torch.load("/home/yaroslav/Desktop/DiffusionIK/weights/seed_epsilon_ik_5_deg_model_30.pth",
                                 map_location=torch.device('cuda')))
model.eval()

dataset = SeedEpsilonIKDataset(torch.device('cuda'), 1, 1, config)
target_pose, seed_joints, d = dataset[0]
n = config.linspace_size
target_pose = target_pose.repeat(n, 1).to(torch.device('cuda'))
seed_joints = seed_joints.repeat(n, 1).to(torch.device('cuda'))
diff = torch.linspace(0, config.max_seed_dist, n).view(-1, 1).to(torch.device('cuda'))
model_result = model(target_pose, seed_joints, diff)

fk = FK(torch.device('cuda'))
pred_pose_R, pred_pose_t = fk(model_result)
target_pose_R, target_pose_t = target_pose[:, :9].view(-1, 3, 3), target_pose[:, 9:].view(-1, 3)
affine_loss = AffineLoss(alpha=1.0, beta=1.0)
loss = affine_loss.loss_fn((pred_pose_R, pred_pose_t), (target_pose_R, target_pose_t))

d_ = diff[torch.where(loss < config.upper_bound)][0]

plt.plot(diff.cpu().detach().numpy(), loss.cpu().detach().numpy())

d = d.cpu().detach().numpy()
# plt.axvline(x=d, color='r')

d_with_min_loss = diff[loss.argmin()].cpu().detach().numpy()
# plt.axvline(x=d_.cpu().detach().numpy(), color='g')

plt.show()