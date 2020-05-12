#### baseline:fasterrcnn50_rpn
| lr | epoch | bs | lr_scheduler | train_size | precision |
| :----:| :----: | :----: | :----: | :----: | :----: |
| 0.005 | 12 | 8 |  StepLR step_size=3, gamma=0.1 | 1333*800 | 0.6354 |
| 0.0025 | 12 | 8 | StepLR step_size=3, gamma=0.1 |   1333*800 |0.6410 |
| 0.0025 | 20 | 8 | MultiStepLR milestones=[16,19], gamma=0.1 | 1333*800 | 0.6548 |
| 0.0025 | 30 | 8 | MultiStepLR milestones=[16,19], gamma=0.1 | 1333*800 | 0.6567 |
| 0.0025 | 20 | 8 | MultiStepLR milestones=[16,19], gamma=0.1 | 1024*1024 | 0.6595 |
| 0.0025 | 20 | 8 | MultiStepLR milestones=[16,19], gamma=0.1 | 1024*1024+norm | 0.6647 |
| 0.01 |30 | 8 | StepLR(optimizer, step_size=5, gamma=0.5) weight_decay 0.0001 |1024*1024+norm |0.6704|

default:
* optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)