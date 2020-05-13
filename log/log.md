#### baseline:fasterrcnn50_rpn
| lr | epoch | bs | lr_scheduler | train_size | precision |
| :----:| :----: | :----: | :----: | :----: | :----: |
| 0.005 | 12 | 8 |  StepLR step_size=3, gamma=0.1 | 1333*800 | 0.6354 |
| 0.0025 | 12 | 8 | StepLR step_size=3, gamma=0.1 |   1333*800 |0.6410 |
| 0.0025 | 20 | 8 | MultiStepLR milestones=[20,29], gamma=0.1 | 1333*800 | 0.6548 |
| 0.0025 | 30 | 8 | MultiStepLR milestones=[20,29], gamma=0.1 | 1333*800 | 0.6567 |
| 0.0025 | 20 | 8 | MultiStepLR milestones=[20,29], gamma=0.1 | 1024*1024 | 0.6595 |
| 0.0025 | 20 | 8 | MultiStepLR milestones=[20,29], gamma=0.1 | 1024*1024+norm | 0.6647 |
| 0.01 |30 | 8 | StepLR(optimizer, step_size=5, gamma=0.5) weight_decay 0.0001 |1024*1024+norm(wheat)|0.6390|
| 0.01 |30 | 8 | StepLR(optimizer, step_size=10, gamma=0.5) weight_decay 0.0001 |1024*1024+norm |0.6292|
| 0.01 |30 | 8 | StepLR(optimizer, step_size=5, gamma=0.5) weight_decay 0.0001 |1024*1024+norm |0.6704|

#### baseline:fasterrcnn50_rpn , Pseudo Labeling retrain
| lr | epoch | bs | lr_scheduler | train_size | pretrainmode |precision |
| :----:| :----: | :----: | :----: | :----: | :----: | :----: |
| 0.01 |10 | 8 | StepLR(optimizer, step_size=5, gamma=0.5) weight_decay 0.0001 |1024*1024+norm(wheat)| 0.6704 mode |0.6753|
| 0.01 |20 | 8 | StepLR(optimizer, step_size=5, gamma=0.5) weight_decay 0.0001 |1024*1024+norm(wheat)| 0.6704 mode |0.6821|
| 0.01 |30 | 8 | StepLR(optimizer, step_size=5, gamma=0.5) weight_decay 0.0001 |1024*1024+norm(wheat)| 0.6704 mode |0.6843|
| 0.01 |10 | 8 | StepLR(optimizer, step_size=5, gamma=0.5) weight_decay 0.0001 |1024*1024+norm(wheat)| 0.6843 mode |0.6711|
default:

* optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)