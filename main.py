from compare_models_file.LSTM import LSTM
from compare_models_file.vallian_attention.Transformer import Transformer
from Model_FAN.FAN import FAN
from utils.ModelController import ModelController
from setting import FAN_SETTING, LSTM_SETTING, VANILLAT_TRANSFORMER_SETTING
from torch.optim import Adam, AdamW, Adamax, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
from datetime import datetime
import time
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import json

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

class Train_Val_Pipeline(object):
    def __init__(self, model:nn.Module, model_setting:dict):
        self.dataset_setting = model_setting['DATASET']
        self.optimizer_setting = model_setting['OPTIMIZER']
        self.model_args = model_setting['MODEL_HYPERPARAMETERS']
        
        print('-> model init...')
        if issubclass(model, nn.Module) and isinstance(self.model_args, dict):
            self.model = model(**(model_setting['MODEL_HYPERPARAMETERS']))
        else:
            raise ValueError(f'''
            Train_Val_Pipeline.model_args must be dict, got {self.model_args.__class__} instead.
            Train_Val_Pipeline.model must be nn.Module, got {model.__class__} instead.
            ''')
        print('-> model init complete')
        time.sleep(0.5)
        print('-> optimizer init...')
        if isinstance(self.optimizer_setting, dict):
            args = self.optimizer_setting['ARGS']
            if self.optimizer_setting['NAME'] in ['AdamW', 'adamW', 'adamw', 'ADAMW']:
                self.optimizer = AdamW(
                    params=self.model.parameters(),
                    lr = args['LR'],
                    betas = args['BETAS']
                )
                if "LR_SCHEDULER" in self.optimizer_setting.keys():
                    print('->-> lr scheduler init...')
                    self.lr_scheduler = self.getLrScheduler()
                    print('->-> lr scheduler init complete')
            
            elif self.optimizer_setting['NAME'] in ['Adam', 'adam', 'ADAM']:
                self.optimizer = Adam(
                    params=self.model.parameters(),
                    lr = args['LR'],
                    betas = args['BETAS']
                )
                if "LR_SCHEDULER" in self.optimizer_setting.keys():
                    print('->-> lr scheduler init...')
                    self.lr_scheduler = self.getLrScheduler()
                    print('->-> lr scheduler init complete')
            
            elif self.optimizer_setting['NAME'] in ['Adamax', 'adamax', 'ADAMAX']:
                self.optimizer = Adamax(
                    params=self.model.parameters(),
                    lr = args['LR'],
                    betas = args['BETAS']
                )
                if "LR_SCHEDULER" in self.optimizer_setting.keys():
                    print('->-> lr scheduler init complete')
                    self.lr_scheduler = self.getLrScheduler()
                    print('->-> lr scheduler init complete')
            
            elif self.optimizer_setting['NAME'] in ['SGD', 'sgd', 'SGD']:
                self.optimizer = SGD(
                    params=self.model.parameters(),
                    lr = args['LR'],
                    momentum = args['MOMENTUM'],
                )
                if "LR_SCHEDULER" in self.optimizer_setting.keys():
                    print('->-> lr scheduler init complete')
                    self.lr_scheduler = self.getLrScheduler()
                    print('->-> lr scheduler init complete')
            
            else:
                raise ValueError(f'''
                Train_Val_Pipeline.optimizer_setting['NAME'] must be "AdamW", "Adam", "SGD", got {self.optimizer_setting['NAME']} instead.
                ''')
        print('-> optimizer init complete')
        time.sleep(0.5)
        print('-> dataset init...')

        if self.dataset_setting['MODE'] == 'stimulation':
            if self.dataset_setting['OPTION_TYPE'] in ['euro', 'european', 'EURO', 'Euro', 'European']:
                self.model_controller = ModelController(option_type = 'euro')
            elif self.dataset_setting['OPTION_TYPE'] in ['asia', 'Asian', 'ASIA', 'Asia', 'Asian']:
                self.model_controller = ModelController(option_type = 'asia')
            else:
                raise ValueError(f'''
                Train_Val_Pipeline.dataset_setting['OPTION_TYPE'] must be "euro", "european", "EURO", "Euro", "European", "asia", "Asian", "ASIA", "Asia", "Asian", got {self.dataset_setting['OPTION_TYPE']} instead.
                ''')
        else:
            if self.dataset_setting['OPTION_TYPE'] in ['euro', 'european', 'EURO', 'Euro', 'European']:
                self.model_controller = ModelController(option_type = 'euro', dataset_name = self.dataset_setting['PATH'])
            elif self.dataset_setting['OPTION_TYPE'] in ['asia', 'Asian', 'ASIA', 'Asia', 'Asian']:
                self.model_controller = ModelController(option_type = 'asia', dataset_name = self.dataset_setting['PATH'])
            else:
                raise ValueError(f'''
                Train_Val_Pipeline.dataset_setting['OPTION_TYPE'] must be "euro", "european", "EURO", "Euro", "European", "asia", "Asian", "ASIA", "Asia", "Asian", got {self.dataset_setting['OPTION_TYPE']} instead.
                ''')
        
        print('-> dataset init complete')

        try:
            a = self.model
            b = self.optimizer
            c = self.model_controller
            if "LR_SCHEDULER" in self.optimizer_setting.keys():
                d = self.lr_scheduler
        except:
            raise ValueError(f'''
            self.model, self.optimizer, self.model_controller code is wrong, checking please!
            ''')              
    
    def getLrScheduler(self):
        lr_scheduler_args = self.optimizer_setting['LR_SCHEDULER']
        if self.optimizer_setting['LR_SCHEDULER']['NAME'] in ['CosineAnnealingLR', 'cosineannealinglr', 'COSA']:
            try:
                lr_scheduler = CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max = lr_scheduler_args['T_MAX'],
                    eta_min = lr_scheduler_args['ETA_MIN'],
                )
                return lr_scheduler
            except KeyError:
                raise ValueError(f'''
                Train_Val_Pipeline.optimizer_setting['LR_SCHEDULER']['T_MAX'] and Train_Val_Pipeline.optimizer_setting['LR_SCHEDULER']['ETA_MIN'] must be set.
                ''')
        elif self.optimizer_setting['LR_SCHEDULER']['NAME'] in ['StepLR', 'steplr', 'STEP']:
            try:
                lr_scheduler = StepLR(
                    optimizer=self.optimizer,
                    step_size = lr_scheduler_args['STEP_SIZE'],
                    gamma = lr_scheduler_args['GAMMA'],
                )
                return lr_scheduler
            except KeyError:
                raise ValueError(f'''
                Train_Val_Pipeline.optimizer_setting['LR_SCHEDULER']['STEP_SIZE'] and Train_Val_Pipeline.optimizer_setting['LR_SCHEDULER']['GAMMA'] must be set.
                ''')
        elif self.optimizer_setting['LR_SCHEDULER']['NAME'] in ['CosineAnnealingWarmRestarts', 'cosineannealingwarmrestarts', 'COSAWARM', 'cosawarm', 'COSWARM', 'coswarm']:
            try:
                lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0 = lr_scheduler_args['T_0'],
                T_mult = lr_scheduler_args['T_MULT'],
                eta_min = lr_scheduler_args['ETA_MIN'],
                )
                return lr_scheduler
            except KeyError:
                raise ValueError(f'''
                Train_Val_Pipeline.optimizer_setting['LR_SCHEDULER']['T_0'], Train_Val_Pipeline.optimizer_setting['LR_SCHEDULER']['T_MULT'], and Train_Val_Pipeline.optimizer_setting['LR_SCHEDULER']['ETA_MIN'] must be set.
                ''')
    
    def run(self, continue_train=False, save_model_file_name:str='model.pth', is_plot:bool=True):
        model = self.model
        if continue_train:
            model.load_state_dict(
                torch.load(
                    f"{ABS_PATH}\\Trained_Model\\{save_model_file_name}_torch_state_dict.pth"
                )
            )
        
        training_loss, testing_loss, model = self.model_controller.trainingModel(
                                                model = model,
                                                Optimizer = self.optimizer,
                                                lr_scheduler = self.lr_scheduler,
                                                epochs = self.optimizer_setting['EPOCHS'],
                                                batch_size = self.dataset_setting['BATCH_SIZE'],
                                            )

        torch.save(
            model.state_dict(),
            f"{ABS_PATH}\\Trained_Model\\{save_model_file_name}.pth"
        )

        if is_plot:
            plt.figure(figsize=(10, 5), dpi=300)
            plt.plot(training_loss)
            plt.plot(testing_loss)
            plt.legend()
            plt.savefig(f"{ABS_PATH}\\Loss_Plot\\{save_model_file_name}.pdf")
            plt.show()
        
        self.model_report = self.modelReport(model = model, model_name = save_model_file_name)
    
    def modelReport(self, model:nn.Module, model_name:str):
        model.eval()
        validation_loss_mean, validation_loss_std, delta_pair = self.model_controller.validationModel(model = model)
        
        for pair in delta_pair[:20]:
            plt.plot(pair[0].detach().cpu().numpy(), label='pred')
            plt.plot(pair[1].detach().cpu().numpy(), label='target')
            plt.legend()
            plt.show()


        info = {
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Name': model_name,
            'Hyperparameters': self.model_args,
            'Dataset': self.dataset_setting,
            'Optimizer': self.optimizer_setting,
            'LR_scheduler': self.optimizer_setting['LR_SCHEDULER'],
            'Epochs': self.optimizer_setting['EPOCHS'],
            'Batch Size': self.dataset_setting['BATCH_SIZE'],
            'Validation Loss Mean': float(validation_loss_mean),
            'Validation Loss Std': float(validation_loss_std),
        }

        report_dir = os.path.join(ABS_PATH, 'TrainingReport')
        os.makedirs(report_dir, exist_ok=True)  # 确保目录存在
    
        # 生成文件名（包含时间戳和模型名称）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = f"{model_name}_{timestamp}_report.json"
        json_filepath = os.path.join(report_dir, json_filename)
    
        # 保存JSON文件
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
        print(f"训练报告已保存至: {json_filepath}")
        return info
        
if __name__ == '__main__':
    # 训练加测试的代码
    pipeline1 = Train_Val_Pipeline(model=FAN, model_setting=FAN_SETTING)
    pipeline1.run(continue_train=False, save_model_file_name='FAN_test', is_plot=True)


    # 不训练，只测试的代码
    pipeline1 = Train_Val_Pipeline(model=FAN, model_setting=FAN_SETTING)
    pipeline1.model.load_state_dict(torch.load(f"{ABS_PATH}\\Trained_Model\\FAN_test.pth"), weights_only=True)
    pipeline1.modelReport(model = pipeline1.model.to('cuda'), model_name = 'FAN_test')

