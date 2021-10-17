
import wandb
def get_sweep_id(method):
    sweep_config = {
        'method': method
        , 'metric' : {'name':'val_loss', 'goal':'minimize'}, 
        'parameters' : {
            
            'batch_size':{
                'values':[3, 4, 5]
            },

            # },
            # 'gradient_accumulation_steps':{
            #     'values':[1,2]
            # },
            'lr':{
                'values':[1e-5,5e-5,2e-6]
            },
            'weight_decay':{
                'values':[0.0,1e-5]
            },
            # 'adam_epsilon':{
            #     'values':[1e-7,1e-8,1e-7]
            # },
            'max_grad_norm':{
                'values':[0.5,1.0,1.5]
            },
            'temperature':{
                'value':[0.1,0.3,0.5]
            },
            # 'warmup_steps':{
            #     'values':[0,1]
            # },
            'threshold':{
                'value':[0.1, 0.3,0.5]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config) 
    return sweep_id

