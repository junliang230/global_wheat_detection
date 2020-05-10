#### baseline:fasterrcnn50_rpn
| lr | epoch | bs | lr_scheduler | precision |
| :----:| :----: | :----: | :----: | :----: |
| 0.005 | 12 | 8 |  StepLR step_size=3, gamma=0.1 | 0.6354 |
| 0.0025 | 12 | 8 | StepLR step_size=3, gamma=0.1 | 0.6410 |
| 0.0025 | 20 | 8 | MultiStepLR milestones=[16,19], gamma=0.1 | 0.6548 |
| 0.0025 | 30 | 8 | MultiStepLR milestones=[20,29], gamma=0.1 | 0.6567 |


default:
* optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)