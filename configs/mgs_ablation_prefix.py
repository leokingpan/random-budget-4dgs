# Full-loss ablation: Prefix only (mgs_full_weight=0.0)
# Based on mgs_4dgs.py for bouncingballs
sh_degree = 3
source_path = 'data/dnerf/bouncingballs'
model_path = './output/dnerf/bouncingballs_ablation_prefix'
images = 'images'; resolution = -1; white_background = True; data_device = 'cuda'; eval = True
iterations = 30_000; coarse_iterations = 3_000
position_lr_init = 0.00016; position_lr_final = 0.0000016
deformation_lr_init = 0.00016; deformation_lr_final = 0.000016
grid_lr_init = 0.0016; grid_lr_final = 0.00016
feature_lr = 0.0025; opacity_lr = 0.05; scaling_lr = 0.005; rotation_lr = 0.001
percent_dense = 0.01; lambda_dssim = 0.2
densification_interval = 100; densify_from_iter = 500; densify_until_iter = 15_000
opacity_reset_interval = 3000

ModelHiddenParams = dict(
    net_width=64, defor_depth=1, bounds=1.6,
    plane_tv_weight=0.0001, time_smoothness_weight=0.01,
    l1_time_planes=0.0001,
    kplanes_config={'grid_dimensions': 2, 'input_coordinate_dim': 4,
                    'output_coordinate_dim': 32, 'resolution': [64, 64, 64, 75]},
    multires=[1, 2, 4, 8],
    no_dx=False, no_grid=False, no_ds=False, no_dr=False,
    no_do=True, no_dshs=True,
    empty_voxel=False, grid_pe=0, static_mlp=False, apply_rotation=False,
)

ip = '127.0.0.1'; port = 6041; quiet = False
mgs_enabled = True; mgs_ratio = 0.5; mgs_full_weight = 0.0
budget_min = 0.1; budget_max = 1.0
ordering_strategy = 'opacity_area'
expname = 'dnerf/bouncingballs_ablation_prefix'
